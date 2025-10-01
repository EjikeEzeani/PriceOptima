from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    # Read and parse CSV
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    headers = list(df.columns)
    rows = df.to_dict(orient="records")
    summary = {
        "totalRecords": len(df),
        "dateRange": f"{df[headers[0]].iloc[0] if len(df) else 'N/A'} to {df[headers[0]].iloc[-1] if len(df) else 'N/A'}",
        "products": df[headers[1]].nunique() if len(headers) > 1 else 0,
        "categories": df[headers[2]].nunique() if len(headers) > 2 else 0,
        "totalRevenue": float(df[headers[5]].sum()) if len(headers) > 5 else 0,
        "avgPrice": float(df[headers[3]].mean()) if len(headers) > 3 else 0,
    }
    preview = rows[:5]
    return {
        "files": [{"name": file.filename, "size": file.size, "type": file.content_type}],
        "headers": headers,
        "rows": rows[:1000],
        "summary": summary,
        "preview": preview,
    }

@app.post("/eda")
async def run_eda(request: Request):
    body = await request.json()
    # For now, return a mock EDA result
    return {
        "overview": {
            "category_distribution": {"Vegetables": 40, "Grains": 30, "Dairy": 20, "Other": 10},
            "revenue_vs_waste": {"revenue": [1000, 1200, 900], "waste": [50, 60, 40]},
        },
        "trends": {
            "sales_over_time": [100, 120, 130, 110, 140],
        },
        "correlations": {
            "price_quantity": 0.65,
            "price_revenue": 0.72,
        },
        "insights": [
            "Peak sales occur on weekends with 35% higher volume",
            "Dairy products show highest waste percentage at 18%",
            "Price elasticity varies significantly across categories",
        ],
        "recommendations": [
            "Implement dynamic pricing for high-waste categories",
            "Focus ML models on perishable goods optimization",
            "Consider seasonal adjustments in pricing strategy",
        ],
    }

@app.post("/ml")
async def run_ml(request: Request):
    body = await request.json()
    model = body.get("model", "xgboost")
    # Return a mock ML result
    return {
        "modelId": model,
        "metrics": {
            "r2": 0.91 if model == "xgboost" else 0.85 if model == "rf" else 0.78,
            "rmse": 245.3 if model == "xgboost" else 298.7 if model == "rf" else 356.2,
            "mae": 189.4 if model == "xgboost" else 234.1 if model == "rf" else 287.9,
        },
        "predictions": [
            {"actual": 1000 + i * 50, "predicted": 1000 + i * 50 + 20, "product": f"Product {i+1}"} for i in range(10)
        ],
        "featureImportance": [
            {"feature": "Historical Sales", "importance": 0.35},
            {"feature": "Seasonality", "importance": 0.28},
            {"feature": "Price", "importance": 0.22},
            {"feature": "Day of Week", "importance": 0.15},
        ],
    }

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
    body = await request.json()
    items = body.get("items", [])
    # Return a mock export result
    return {"status": "success", "exported": items, "downloadLinks": [f"/downloads/{item}.zip" for item in items]}
