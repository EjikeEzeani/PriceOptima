#!/usr/bin/env bash
set -e

# Integration test script for upload endpoint
# Usage: API_BASE_URL=http://127.0.0.1:8000 ./tests/integration/upload_smoke.sh

URL=${API_BASE_URL:-http://127.0.0.1:8000}

echo "Testing upload endpoint at: $URL"

# Test health endpoint first
echo "Testing health endpoint..."
curl -s "$URL/health" | grep -q "healthy" || {
    echo "❌ Health check failed"
    exit 1
}
echo "✅ Health check passed"

# Create a test CSV file
TEST_CSV="tests/fixtures/sample.csv"
mkdir -p tests/fixtures

cat > "$TEST_CSV" << EOF
Date,Product Name,Category,Price,Quantity Sold,Revenue
2024-01-01,Rice 5kg,Grains,2500,45,112500
2024-01-01,Tomatoes 1kg,Vegetables,800,120,96000
2024-01-02,Bread 1kg,Bakery,300,200,60000
2024-01-02,Milk 1L,Dairy,500,150,75000
2024-01-03,Eggs 12pcs,Dairy,400,100,40000
EOF

echo "Created test CSV file: $TEST_CSV"

# Test upload endpoint
echo "Testing upload endpoint..."
RESPONSE=$(curl -s -w "\n%{http_code}" -F "file=@$TEST_CSV" "$URL/upload")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Upload test passed (HTTP $HTTP_CODE)"
    echo "Response preview:"
    echo "$RESPONSE_BODY" | head -c 200
    echo "..."
else
    echo "❌ Upload test failed (HTTP $HTTP_CODE)"
    echo "Response: $RESPONSE_BODY"
    exit 1
fi

# Test invalid file type
echo "Testing invalid file type..."
INVALID_RESPONSE=$(curl -s -w "\n%{http_code}" -F "file=@tests/fixtures/sample.csv;filename=test.txt" "$URL/upload")
INVALID_HTTP_CODE=$(echo "$INVALID_RESPONSE" | tail -n1)

if [ "$INVALID_HTTP_CODE" = "400" ]; then
    echo "✅ Invalid file type test passed (HTTP $INVALID_HTTP_CODE)"
else
    echo "❌ Invalid file type test failed (HTTP $INVALID_HTTP_CODE)"
    exit 1
fi

echo "🎉 All integration tests passed!"
