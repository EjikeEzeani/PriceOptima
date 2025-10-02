#!/usr/bin/env python3
"""
Lightweight backend for upload testing - no heavy ML libraries
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dynamic Pricing Analytics API - Lightweight")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data storage
uploaded_data = None
processed_data = None

# Create exports directory
os.makedirs("exports", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Dynamic Pricing Analytics API - Lightweight", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running", "timestamp": datetime.now().isoformat()}

def validate_data(df):
    """Validate uploaded data and ensure it has required columns"""
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Check for required columns with flexible matching
    required_columns = ['price', 'quantity', 'revenue']
    missing_columns = []
    column_mapping = {}
    
    # First check for exact matches
    for req_col in required_columns:
        if req_col in df.columns:
            column_mapping[req_col] = req_col
        else:
            missing_columns.append(req_col)
    
    # If we have missing columns, try flexible matching
    if missing_columns:
        logger.info(f"Looking for flexible column matches for: {missing_columns}")
        
        for req_col in missing_columns:
            found_match = False
            
            # Try different matching strategies
            for col in df.columns:
                col_lower = col.lower().strip()
                req_lower = req_col.lower().strip()
                
                # Exact match (case insensitive)
                if col_lower == req_lower:
                    column_mapping[req_col] = col
                    found_match = True
                    break
                
                # Contains match
                elif req_lower in col_lower or col_lower in req_lower:
                    column_mapping[req_col] = col
                    found_match = True
                    break
                
                # Common variations
                elif (req_col == 'price' and any(x in col_lower for x in ['price', 'cost', 'amount', 'value'])) or \
                     (req_col == 'quantity' and any(x in col_lower for x in ['quantity', 'qty', 'amount', 'count', 'units', 'volume'])) or \
                     (req_col == 'revenue' and any(x in col_lower for x in ['revenue', 'sales', 'income', 'total', 'earnings'])):
                    column_mapping[req_col] = col
                    found_match = True
                    break
            
            if not found_match:
                missing_columns.append(req_col)
    
    # Apply column mapping
    if column_mapping:
        logger.info(f"Column mapping applied: {column_mapping}")
        for req_col, actual_col in column_mapping.items():
            if req_col != actual_col:
                df[req_col] = df[actual_col]
    
    # Check if we still have missing columns
    still_missing = [col for col in required_columns if col not in df.columns]
    if still_missing:
        available_columns = list(df.columns)
        raise ValueError(f"Missing required columns: {still_missing}. Available columns: {available_columns}")
    
    # Ensure numeric columns are numeric
    for col in required_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().all():
                raise ValueError(f"Column '{col}' contains no valid numeric data")
    
    logger.info(f"Data validation successful. Columns: {list(df.columns)}")
    return df

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload and process data file - lightweight version"""
    try:
        global uploaded_data, processed_data
        
        logger.info(f"Uploading file: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Parse CSV
        df = pd.read_csv(io.BytesIO(content))
        
        # Validate data
        df = validate_data(df)
        
        # Store data globally
        uploaded_data = df
        processed_data = df.copy()
        
        # Generate summary efficiently
        headers = list(df.columns)
        
        # Find columns by name instead of index (optimized)
        date_col = None
        product_col = None
        category_col = None
        price_col = None
        revenue_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['date', 'time', 'day']):
                date_col = col
            elif any(x in col_lower for x in ['product', 'item', 'name']):
                product_col = col
            elif any(x in col_lower for x in ['category', 'type', 'class']):
                category_col = col
            elif any(x in col_lower for x in ['price', 'cost', 'amount']):
                price_col = col
            elif any(x in col_lower for x in ['revenue', 'sales', 'total']):
                revenue_col = col
        
        # Calculate summary efficiently using vectorized operations
        summary = {
            "totalRecords": len(df),
            "dateRange": f"{df[date_col].iloc[0] if date_col and len(df) else 'N/A'} to {df[date_col].iloc[-1] if date_col and len(df) else 'N/A'}",
            "products": df[product_col].nunique() if product_col else 0,
            "categories": df[category_col].nunique() if category_col else 0,
            "totalRevenue": float(df[revenue_col].sum()) if revenue_col else 0,
            "avgPrice": float(df[price_col].mean()) if price_col else 0,
        }
        
        # Only generate preview data (5 rows) for immediate display
        preview = df.head(5).to_dict(orient="records")
        
        logger.info(f"Data uploaded successfully - {len(df)} records")
        
        return {
            "files": [{"name": file.filename, "size": file.size, "type": file.content_type}],
            "headers": headers,
            "rows": [],  # Don't send all rows immediately - use pagination
            "summary": summary,
            "preview": preview,
            "status": "success",
            "message": f"Successfully uploaded {len(df)} records",
            "totalRows": len(df)  # Add total count for pagination
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.get("/data/rows")
async def get_data_rows(page: int = 0, limit: int = 100):
    """Get paginated data rows for display"""
    try:
        global uploaded_data
        
        if uploaded_data is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        start_idx = page * limit
        end_idx = start_idx + limit
        
        # Get paginated data
        paginated_data = uploaded_data.iloc[start_idx:end_idx]
        rows = paginated_data.to_dict(orient="records")
        
        return {
            "rows": rows,
            "page": page,
            "limit": limit,
            "totalRows": len(uploaded_data),
            "hasMore": end_idx < len(uploaded_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get data rows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get data rows: {str(e)}")

# Placeholder endpoints for compatibility
@app.post("/eda")
async def run_eda():
    """Placeholder EDA endpoint"""
    return {"message": "EDA not available in lightweight mode", "status": "disabled"}

@app.post("/ml")
async def run_ml():
    """Placeholder ML endpoint"""
    return {"message": "ML not available in lightweight mode", "status": "disabled"}

@app.post("/rl")
async def run_rl():
    """Placeholder RL endpoint"""
    return {"message": "RL not available in lightweight mode", "status": "disabled"}

@app.post("/export")
async def export_data():
    """Placeholder export endpoint"""
    return {"message": "Export not available in lightweight mode", "status": "disabled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
