from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import io
import json
import os
import csv
from datetime import datetime
import gc  # For garbage collection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PriceOptima API - Super Minimal", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data storage
uploaded_data = None
processed_data = None
eda_results = None

# Memory optimization: limit data size
MAX_ROWS = 1000  # Limit to 1k rows max
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB max file size

def clear_memory():
    """Clear memory and run garbage collection"""
    global uploaded_data, processed_data, eda_results
    uploaded_data = None
    processed_data = None
    eda_results = None
    gc.collect()

def parse_csv_simple(content):
    """Parse CSV using only built-in csv module"""
    try:
        # Decode bytes to string
        text = content.decode('utf-8')
        lines = text.split('\n')
        
        # Parse CSV manually
        reader = csv.reader(lines)
        rows = list(reader)
        
        if not rows:
            return []
        
        # First row is headers
        headers = rows[0]
        data_rows = rows[1:]
        
        # Convert to list of dictionaries
        result = []
        for row in data_rows:
            if len(row) == len(headers):  # Skip incomplete rows
                row_dict = {}
                for i, header in enumerate(headers):
                    row_dict[header] = row[i]
                result.append(row_dict)
        
        return result[:MAX_ROWS]  # Limit rows
        
    except Exception as e:
        logger.error(f"CSV parsing failed: {e}")
        return []

def calculate_basic_stats(data):
    """Calculate basic statistics without pandas"""
    if not data:
        return {}
    
    # Get numeric columns
    numeric_cols = []
    for key in data[0].keys():
        try:
            float(data[0][key])
            numeric_cols.append(key)
        except (ValueError, TypeError):
            continue
    
    stats = {}
    for col in numeric_cols:
        values = []
        for row in data:
            try:
                values.append(float(row[col]))
            except (ValueError, TypeError):
                continue
        
        if values:
            stats[col] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
    
    return stats

@app.get("/")
async def root():
    return {"message": "PriceOptima API - Super Minimal", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running", "timestamp": datetime.now().isoformat()}

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload and process data with memory optimization"""
    try:
        global uploaded_data, processed_data, eda_results
        
        # Clear previous data to free memory
        clear_memory()
        
        # Check file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max size is {MAX_FILE_SIZE // (1024*1024)}MB")
        
        # Parse CSV using built-in csv module
        data = parse_csv_simple(content)
        
        if not data:
            raise HTTPException(status_code=400, detail="Failed to parse CSV file")
        
        # Store data
        uploaded_data = data
        processed_data = data.copy()
        eda_results = None
        
        # Generate summary
        summary = {
            "totalRecords": len(data),
            "columns": list(data[0].keys()) if data else [],
            "memoryUsage": f"{len(str(data)) / 1024 / 1024:.2f} MB"
        }
        
        logger.info(f"Data uploaded successfully - {len(data)} records")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(data)} records",
            "summary": summary,
            "preview": data[:5]  # First 5 rows
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        clear_memory()  # Clear memory on error
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.post("/eda")
async def run_eda():
    """Run basic EDA analysis"""
    try:
        global uploaded_data, eda_results
        
        if uploaded_data is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        # Basic EDA using built-in functions
        data = uploaded_data.copy()
        
        # Calculate basic statistics
        stats = calculate_basic_stats(data)
        
        eda_results = {
            "overview": {
                "total_records": len(data),
                "columns": list(data[0].keys()) if data else [],
                "numeric_columns": len(stats)
            },
            "basic_stats": stats,
            "insights": [
                f"Dataset contains {len(data)} records",
                f"Found {len(stats)} numeric columns"
            ],
            "recommendations": [
                "Consider data sampling for large datasets",
                "Monitor memory usage during analysis"
            ]
        }
        
        # Clear memory after processing
        gc.collect()
        
        return eda_results
        
    except Exception as e:
        logger.error(f"EDA failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")

@app.post("/ml")
async def run_ml(request: Request):
    """Mock ML training - no heavy computation"""
    try:
        global uploaded_data
        
        if uploaded_data is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        # Mock ML results to avoid heavy computation
        result = {
            "modelId": "mock_linear_regression",
            "metrics": {
                "r2": 0.85,
                "rmse": 0.12,
                "mae": 0.08
            },
            "predictions": [
                {"actual": 100.0, "predicted": 95.0},
                {"actual": 120.0, "predicted": 115.0},
                {"actual": 90.0, "predicted": 88.0}
            ],
            "featureImportance": [
                {"feature": "price", "importance": 0.6},
                {"feature": "quantity", "importance": 0.4}
            ]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"ML training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML training failed: {str(e)}")

@app.post("/rl")
async def run_rl(request: Request):
    """Mock RL simulation - no heavy computation"""
    return {
        "algorithm": "dqn",
        "finalReward": 100.0,
        "convergenceEpisode": 500,
        "policy": {
            "wasteReduction": 20.0,
            "profitIncrease": 15.0,
            "customerSatisfaction": 0.85
        },
        "trainingCurve": [
            {"episode": i, "reward": 50 + i * 0.1}
            for i in range(0, 500, 10)
        ]
    }

@app.post("/export")
async def export_results(request: Request):
    """Lightweight export - JSON only"""
    try:
        global uploaded_data, eda_results
        
        if uploaded_data is None:
            raise HTTPException(status_code=400, detail="No data available")
        
        # Create exports directory
        os.makedirs("exports", exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = []
        
        # Export only essential data as JSON
        export_data = {
            "timestamp": timestamp,
            "data_summary": {
                "total_records": len(uploaded_data),
                "columns": list(uploaded_data[0].keys()) if uploaded_data else []
            },
            "eda_results": eda_results
        }
        
        # Save as JSON only (lightweight)
        json_path = f"exports/export_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        exported_files.append(os.path.basename(json_path))
        
        return {
            "status": "success",
            "exported": ["summary_report"],
            "files": exported_files,
            "message": f"Exported {len(exported_files)} files"
        }
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download exported file"""
    file_path = os.path.join("exports", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/status")
async def get_status():
    """Get current status"""
    return {
        "data_uploaded": uploaded_data is not None,
        "data_records": len(uploaded_data) if uploaded_data is not None else 0,
        "eda_completed": eda_results is not None,
        "memory_optimized": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
