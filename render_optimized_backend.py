from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import io
import json
import os
from datetime import datetime
import numpy as np
import gc  # For garbage collection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PriceOptima API - Memory Optimized", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data storage - use smaller data structures
uploaded_data = None
processed_data = None
eda_results = None

# Memory optimization: limit data size
MAX_ROWS = 10000  # Limit to 10k rows max
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size

def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    try:
        # Limit rows
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            logger.info(f"Limited dataset to {MAX_ROWS} rows for memory optimization")
        
        # Convert object columns to category to save memory
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Downcast numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    except Exception as e:
        logger.error(f"Data optimization failed: {e}")
        return df

def clear_memory():
    """Clear memory and run garbage collection"""
    global uploaded_data, processed_data, eda_results
    uploaded_data = None
    processed_data = None
    eda_results = None
    gc.collect()

@app.get("/")
async def root():
    return {"message": "PriceOptima API - Memory Optimized", "status": "running"}

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
        
        # Parse CSV
        df = pd.read_csv(io.BytesIO(content))
        
        # Optimize memory usage
        df = optimize_dataframe(df)
        
        # Store optimized data
        uploaded_data = df
        processed_data = df.copy()
        eda_results = None
        
        # Generate summary efficiently
        summary = {
            "totalRecords": len(df),
            "columns": list(df.columns),
            "memoryUsage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        
        logger.info(f"Data uploaded successfully - {len(df)} records, {summary['memoryUsage']} memory")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(df)} records",
            "summary": summary,
            "preview": df.head(5).to_dict(orient="records")
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        clear_memory()  # Clear memory on error
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.post("/eda")
async def run_eda():
    """Run lightweight EDA analysis"""
    try:
        global uploaded_data, eda_results
        
        if uploaded_data is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        # Lightweight EDA - avoid heavy computations
        df = uploaded_data.copy()
        
        # Basic statistics only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        eda_results = {
            "overview": {
                "total_records": len(df),
                "columns": list(df.columns),
                "numeric_columns": len(numeric_cols)
            },
            "basic_stats": {
                "mean": df[numeric_cols].mean().to_dict() if numeric_cols else {},
                "std": df[numeric_cols].std().to_dict() if numeric_cols else {}
            },
            "insights": [
                f"Dataset contains {len(df)} records",
                f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
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
    """Lightweight ML training"""
    try:
        global uploaded_data
        
        if uploaded_data is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        # Use only a sample for ML to save memory
        df_sample = uploaded_data.sample(n=min(1000, len(uploaded_data)), random_state=42)
        
        # Simple linear regression only
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise HTTPException(status_code=400, detail="Insufficient numeric data")
        
        X = df_sample[numeric_cols[:-1]]
        y = df_sample[numeric_cols[-1]]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        result = {
            "modelId": "linear_regression",
            "metrics": {"r2": float(r2)},
            "predictions": [
                {"actual": float(actual), "predicted": float(pred)}
                for actual, pred in zip(y_test[:10], predictions[:10])
            ],
            "featureImportance": [
                {"feature": col, "importance": float(abs(coef))}
                for col, coef in zip(X.columns, model.coef_)
            ]
        }
        
        # Clear memory
        del df_sample, X, y, X_train, X_test, y_train, y_test, model
        gc.collect()
        
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
                "columns": list(uploaded_data.columns)
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
