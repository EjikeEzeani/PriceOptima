from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import io
import json
import os
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib
import zipfile
import pandas as pd

app = FastAPI()

# Allow CORS for local frontend dev (explicit origins to avoid credential wildcard issues)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3010",
    "http://127.0.0.1:3010",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data storage (in production, use a database)
uploaded_data = None
processed_data = None
eda_results_cache = None
eda_completed = False

# Constants
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
REQUIRED_COLUMNS_SYNONYMS = {
    "Date": ["date"],
    "Product": ["product", "product name", "product_name", "item", "item name"],
    "Category": ["category", "type"],
    "Price": ["price", "unit price", "unit_price"],
    "Quantity": ["quantity", "quantity sold", "qty", "qty sold"],
    "Revenue": ["revenue", "sales", "total", "total revenue"]
}
OPTIONAL_COLUMNS_SYNONYMS = {
    "Waste": ["waste", "waste amount", "waste_amount"],
    "Cost": ["cost", "unit cost", "unit_cost"],
    "Supplier": ["supplier", "vendor"]
}

# Exports directory (configurable via env var)
EXPORTS_DIR = os.getenv(
    "PRICEOPTIMA_EXPORT_DIR",
    os.path.join(os.path.expanduser("~"), "Documents", "PriceOptima", "exports"),
)
os.makedirs(EXPORTS_DIR, exist_ok=True)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def standardize_columns(original_df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names using synonyms. Returns a copy."""
    df = original_df.copy()
    lower_map = {i: _normalize_column_name(i) for i in df.columns}

    rename_map = {}
    # Required columns
    for standard, synonyms in REQUIRED_COLUMNS_SYNONYMS.items():
        candidates = [standard.lower(), *synonyms]
        match = next((col for col, low in lower_map.items() if low in candidates), None)
        if match is not None:
            rename_map[match] = standard

    # Optional columns
    for standard, synonyms in OPTIONAL_COLUMNS_SYNONYMS.items():
        candidates = [standard.lower(), *synonyms]
        match = next((col for col, low in lower_map.items() if low in candidates), None)
        if match is not None:
            rename_map[match] = standard

    df = df.rename(columns=rename_map)

    # Validate required columns present (after rename)
    missing = [c for c in REQUIRED_COLUMNS_SYNONYMS.keys() if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date column supporting YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY."""
    df = df.copy()
    if "Date" in df.columns:
        # Try multiple formats
        date_series = pd.to_datetime(df["Date"], errors="coerce", format=None)
        # If too many NaT, try dayfirst
        if date_series.isna().mean() > 0.3:
            date_series = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["Date"] = date_series
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preprocess_data(original_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data: standardize columns, parse dates, coerce numerics, compute Revenue when needed, handle missing."""
    df = standardize_columns(original_df)
    df = parse_dates(df)
    df = coerce_numeric(df, ["Price", "Quantity", "Revenue", "Waste", "Cost"])

    # Compute Revenue if missing
    if "Revenue" not in df.columns or df["Revenue"].isna().all():
        if "Price" in df.columns and "Quantity" in df.columns:
            df["Revenue"] = df["Price"].fillna(0) * df["Quantity"].fillna(0)
        else:
            raise HTTPException(status_code=400, detail="Cannot compute Revenue: missing Price or Quantity")

    # Fill missing values
    for col in ["Price", "Quantity", "Revenue", "Waste", "Cost"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Drop rows with missing critical categorical fields; keep rows even if Date is NaT
    # to allow non-time-based analyses to proceed.
    drop_subset = [col for col in ["Product", "Category"] if col in df.columns]
    if drop_subset:
        df = df.dropna(subset=drop_subset).reset_index(drop=True)
    return df


def compute_eda(df: pd.DataFrame) -> dict:
    """Produce summaries, distributions, correlations, outliers."""
    # Variable summaries (guard against empty datasets)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(df) > 0 and numeric_cols:
        try:
            summaries = df[numeric_cols].describe().to_dict()
        except Exception:
            summaries = {}
    else:
        summaries = {}

    # Distributions
    category_distribution = df["Category"].value_counts().to_dict() if "Category" in df.columns else {}

    # Correlations
    correlations = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().fillna(0)
        # expose some key correlations if present
        if all(col in numeric_cols for col in ["Price", "Quantity"]):
            correlations["price_quantity"] = float(corr_matrix.loc["Price", "Quantity"])
        if all(col in numeric_cols for col in ["Price", "Revenue"]):
            correlations["price_revenue"] = float(corr_matrix.loc["Price", "Revenue"])
    
    # Outlier detection using IQR on numeric columns
    outliers = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outliers[col] = {"count": count, "lower": float(lower), "upper": float(upper)}

    return {
        "overview": {
            "category_distribution": category_distribution,
            "revenue_vs_waste": {
                "revenue": df["Revenue"].tolist()[:100] if "Revenue" in df.columns else [],
                "waste": df["Waste"].tolist()[:100] if "Waste" in df.columns else []
            },
        },
        "trends": {
            # Only include rows with valid dates for time-based trend
            "sales_over_time": df.dropna(subset=["Date"]).sort_values("Date")["Revenue"].tolist()[:200] if "Date" in df.columns and "Revenue" in df.columns and len(df) > 0 else []
        },
        "correlations": correlations,
        "summaries": summaries,
        "outliers": outliers,
        "insights": [
            f"Records: {len(df):,}",
            f"Products: {df['Product'].nunique() if 'Product' in df.columns else 0}",
            f"Categories: {df['Category'].nunique() if 'Category' in df.columns else 0}",
        ],
        "recommendations": [
            "Investigate outliers for data quality issues",
            "Use category distribution to focus pricing strategy",
        ],
    }

def process_eda_data(df):
    """Process data for EDA analysis"""
    try:
        # Convert date column if it exists
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate basic statistics
        category_col = None
        for col in df.columns:
            if 'category' in col.lower() or 'type' in col.lower():
                category_col = col
                break
        
        if category_col:
            category_distribution = df[category_col].value_counts().to_dict()
        else:
            category_distribution = {"Unknown": len(df)}
        
        # Calculate correlations
        correlations = {}
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            correlations = {
                "price_quantity": corr_matrix.iloc[0, 1] if len(corr_matrix) > 1 else 0.5,
                "price_revenue": corr_matrix.iloc[0, 2] if len(corr_matrix) > 2 else 0.7
            }
        
        # Generate insights
        insights = [
            f"Dataset contains {len(df)} records with {len(df.columns)} features",
            f"Date range: {df[date_cols[0]].min()} to {df[date_cols[0]].max()}" if date_cols else "No date information available",
            f"Top category: {max(category_distribution, key=category_distribution.get)}" if category_distribution else "No category data"
        ]
        
        recommendations = [
            "Consider implementing dynamic pricing for high-volume categories",
            "Focus on reducing waste in perishable goods categories",
            "Analyze seasonal patterns for better inventory management"
        ]
        
        return {
            "overview": {
                "category_distribution": category_distribution,
                "revenue_vs_waste": {
                    "revenue": df[numeric_cols[0]].tolist()[:10] if numeric_cols else [1000, 1200, 900],
                    "waste": df[numeric_cols[1]].tolist()[:10] if len(numeric_cols) > 1 else [50, 60, 40]
                }
            },
            "trends": {
                "sales_over_time": df[numeric_cols[0]].tolist()[:20] if numeric_cols else [100, 120, 130, 110, 140]
            },
            "correlations": correlations,
            "insights": insights,
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": str(e)}

def train_ml_model(df, model_type="random_forest"):
    """Train ML model on the data"""
    try:
        # Prepare features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {"error": "Insufficient numeric data for ML training"}
        
        X = df[numeric_cols[:-1]]  # All but last column as features
        y = df[numeric_cols[-1]]   # Last column as target
        
        # Select model by type
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=42)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBRegressor  # type: ignore
                model = XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                )
            except Exception:
                # Fallback to Gradient Boosting if XGBoost not available
                model = GradientBoostingRegressor(random_state=42)
        elif model_type == "linear":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

        # Train/test split for meaningful metrics
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        try:
            mape = float(mean_absolute_percentage_error(y_test, predictions) * 100)
        except Exception:
            mape = None
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = [
                {"feature": col, "importance": float(imp)}
                for col, imp in zip(X.columns, model.feature_importances_)
            ]
        else:
            feature_importance = [
                {"feature": col, "importance": 1.0/len(X.columns)}
                for col in X.columns
            ]
        
        results = {
            "modelId": model_type,
            "metrics": {
                "r2": float(r2),
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
            },
            "predictions": [
                {"actual": float(actual), "predicted": float(pred), "product": f"Sample {i+1}"}
                for i, (actual, pred) in enumerate(zip(y_test[:10], predictions[:10]))
            ],
            "featureImportance": feature_importance
        }

        # Persist model artifact
        os.makedirs("models", exist_ok=True)
        artifact_path = os.path.join("models", f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        try:
            joblib.dump(model, artifact_path)
            results["artifactPath"] = artifact_path
        except Exception:
            results["artifactPath"] = None

        return results
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    global uploaded_data, processed_data, eda_results_cache, eda_completed
    try:
        # Validate file type by extension and content type
        filename_lower = (file.filename or "").lower()
        if not filename_lower.endswith(".csv") and (file.content_type or "").lower() != "text/csv":
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")

        # Read bytes and size check
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="File too large. Max size is 50 MB")

        # Parse CSV
        df = pd.read_csv(io.BytesIO(content))
        
        # Preprocess (auto on upload)
        processed = preprocess_data(df)
        uploaded_data = df
        processed_data = processed
        eda_results_cache = None
        eda_completed = False
        
        headers = list(processed.columns)
        rows = processed.to_dict(orient="records")
        
        # Calculate summary statistics
        summary = {
            "totalRecords": len(processed),
            "dateRange": f"{processed['Date'].min()} to {processed['Date'].max()}" if 'Date' in processed.columns and len(processed) else "N/A",
            "products": processed['Product'].nunique() if 'Product' in processed.columns else 0,
            "categories": processed['Category'].nunique() if 'Category' in processed.columns else 0,
            "totalRevenue": float(processed['Revenue'].sum()) if 'Revenue' in processed.columns else 0,
            "avgPrice": float(processed['Price'].mean()) if 'Price' in processed.columns else 0,
        }
        preview = rows[:5]
        
        return {
            "files": [{"name": file.filename, "size": file.size, "type": file.content_type}],
            "headers": headers,
            "rows": rows[:1000],
            "summary": summary,
            "preview": preview,
            "totalRows": len(processed)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to process file: {str(e)}"}
        )

@app.post("/eda")
async def run_eda(request: Request):
    global processed_data, eda_results_cache, eda_completed
    if processed_data is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No data uploaded. Please upload data first."}
        )
    
    try:
        # Compute EDA on processed data
        eda_results_cache = compute_eda(processed_data.copy())
        eda_completed = True
        return eda_results_cache
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"EDA analysis failed: {str(e)}"}
        )

@app.post("/ml")
async def run_ml(request: Request):
    global processed_data, eda_completed
    if processed_data is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No data uploaded. Please upload data first."}
        )
    if not eda_completed:
        return JSONResponse(
            status_code=409,
            content={"error": "EDA must be completed before ML training."}
        )
    
    try:
        body = await request.json()
        model_type = body.get("model", "random_forest")

        # Map frontend model IDs to backend model types
        model_mapping = {
            "linear_regression": "linear",
            "linear": "linear",
            "random_forest": "random_forest",
            "rf": "random_forest",
            "xgboost": "xgboost",
            "gradient_boosting": "gradient_boosting",
            "gb": "gradient_boosting",
        }

        actual_model_type = model_mapping.get(model_type, model_type)
        
        # Train ML model on the processed data
        ml_results = train_ml_model(processed_data.copy(), actual_model_type)
        return ml_results
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"ML training failed: {str(e)}"}
        )

@app.post("/rl")
async def run_rl(request: Request):
    body = await request.json()
    algorithm = body.get("algorithm", "dqn")
    # Return a mock RL result
    return {
        "algorithm": algorithm,
        "finalReward": 123.4,
        "convergenceEpisode": 700,
        "policy": {
            "wasteReduction": 23.5,
            "profitIncrease": 18.2,
            "customerSatisfaction": 0.87,
        },
        "trainingCurve": [
            {"episode": i * 10, "reward": 50 + i * 0.5 + (i % 5) * 2} for i in range(100)
        ],
    }

@app.post("/export")
async def export_results(request: Request):
    global processed_data, eda_results_cache
    if processed_data is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No data available for export. Please upload and analyze data first."}
        )
    
    try:
        body = await request.json()
        items = body.get("items", [])
        
        # Create exports directory if it doesn't exist
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        
        exported_files = []
        
        for item in items:
            if item == "summary_report":
                # Generate summary report
                report_data = {
                    "timestamp": datetime.now().isoformat(),
                    "data_summary": {
                        "total_records": len(processed_data),
                        "columns": list(processed_data.columns),
                        "date_range": "N/A"
                    },
                    "analysis_results": "Analysis completed successfully"
                }
                filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                report_path = os.path.join(EXPORTS_DIR, filename)
                with open(report_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
                exported_files.append(filename)
            
            elif item == "raw_data":
                # Export processed dataset
                csv_filename = f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data_path = os.path.join(EXPORTS_DIR, csv_filename)
                processed_data.to_csv(data_path, index=False)
                exported_files.append(csv_filename)
                # Also export Excel
                try:
                    xlsx_filename = csv_filename.replace('.csv', '.xlsx')
                    xlsx_path = os.path.join(EXPORTS_DIR, xlsx_filename)
                    processed_data.to_excel(xlsx_path, index=False)
                    exported_files.append(xlsx_filename)
                except Exception:
                    pass
            
            elif item == "ml_results":
                # Generate ML results
                ml_results = train_ml_model(processed_data.copy(), "random_forest")
                ml_filename = f"ml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                ml_path = os.path.join(EXPORTS_DIR, ml_filename)
                with open(ml_path, 'w') as f:
                    json.dump(ml_results, f, indent=2)
                exported_files.append(ml_filename)
            elif item == "eda_results":
                if eda_results_cache is None:
                    eda_now = compute_eda(processed_data.copy())
                else:
                    eda_now = eda_results_cache
                eda_filename = f"eda_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                eda_path = os.path.join(EXPORTS_DIR, eda_filename)
                with open(eda_path, 'w') as f:
                    json.dump(eda_now, f, indent=2, default=str)
                exported_files.append(eda_filename)
        
        return {
            "status": "success", 
            "exported": items, 
            "files": exported_files,
            "message": f"Successfully exported {len(exported_files)} files"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Export failed: {str(e)}"}
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(EXPORTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )


@app.get("/download-all")
async def download_all():
    """Create a zip with all exported files and return it."""
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    zip_filename = f"all_exports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(EXPORTS_DIR, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(EXPORTS_DIR):
            for f in files:
                fp = os.path.join(root, f)
                if not fp.endswith('.zip'):
                    zf.write(fp, arcname=os.path.basename(fp))
    return FileResponse(zip_path, filename=zip_filename)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "PriceOptima API is running"}


# Alias endpoints for compatibility with alternate frontends
@app.post("/upload-dataset")
async def upload_dataset_alias(file: UploadFile = File(...)):
    return await upload_data(file)


@app.post("/preprocess")
async def preprocess_alias():
    global processed_data
    if processed_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    return {"status": "ok", "rows": len(processed_data)}


@app.post("/train")
async def train_alias(request: Request):
    return await run_ml(request)


@app.get("/results")
async def results_alias():
    global processed_data, eda_results_cache
    return {
        "uploaded": processed_data is not None,
        "eda_complete": eda_completed,
        "eda": eda_results_cache,
    }


@app.get("/status")
async def status_alias():
    return {
        "uploaded": uploaded_data is not None,
        "processed": processed_data is not None,
        "eda_complete": eda_completed,
        "timestamp": datetime.now().isoformat(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
