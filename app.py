# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Build response
        summary = {
            "totalRecords": len(df),
            "columns": len(df.columns),
            "columnNames": df.columns.tolist()
        }

        preview = df.head(10).to_dict(orient="records")

        return JSONResponse(content={
            "summary": summary,
            "preview": preview
        })

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
