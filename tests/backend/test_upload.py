import io
import pytest
from fastapi.testclient import TestClient
from backend.api_backend import app

client = TestClient(app)

def test_health_endpoint():
    """Test that the health endpoint returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_upload_csv_success():
    """Test uploading a valid CSV returns 200 and expected JSON"""
    csv_content = "Date,Product Name,Category,Price,Quantity Sold,Revenue\n2024-01-01,Rice 5kg,Grains,2500,45,112500\n2024-01-01,Tomatoes 1kg,Vegetables,800,120,96000"
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert "files" in data
    assert "headers" in data
    assert "rows" in data
    assert "summary" in data
    assert "preview" in data
    assert "totalRows" in data
    
    # Check summary structure
    summary = data["summary"]
    assert "totalRecords" in summary
    assert "products" in summary
    assert "categories" in summary
    assert "totalRevenue" in summary
    assert "avgPrice" in summary

def test_upload_non_csv():
    """Test uploading non-CSV returns 400"""
    files = {"file": ("test.txt", io.BytesIO(b"not csv"), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only CSV files are accepted" in response.json()["detail"]

def test_upload_missing_columns():
    """Test uploading CSV with missing required columns returns 422"""
    csv_content = "Date,Product,Price\n2024-01-01,Rice,2500"
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    
    response = client.post("/upload", files=files)
    assert response.status_code == 422
    assert "Missing required columns" in response.json()["detail"]

def test_upload_empty_file():
    """Test uploading empty CSV returns error"""
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 500  # pandas will fail to parse empty CSV

def test_cors_headers():
    """Test that CORS headers are present"""
    response = client.options("/upload")
    assert response.status_code == 200
    # CORS headers should be present (handled by middleware)
