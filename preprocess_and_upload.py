#!/usr/bin/env python3
"""
Preprocess a CSV locally (standardize columns, parse dates, coerce numerics,
compute Revenue if missing), save the cleaned file to a stable folder, and
upload it to the running backend so EDA can proceed immediately.

Usage:
  python preprocess_and_upload.py path/to/your.csv

Output:
  - Cleaned CSV saved to: %USERPROFILE%/Documents/PriceOptima/exports
  - Uploads the (original) file to backend /upload (which also preprocesses)
    so the app can run EDA without "Failed to fetch".
"""
import os
import sys
import re
import pandas as pd
import requests
from datetime import datetime


REQUIRED_COLUMNS_SYNONYMS = {
    "Date": ["date"],
    "Product": ["product", "product name", "product_name", "item", "item name"],
    "Category": ["category", "type"],
    "Price": ["price", "unit price", "unit_price"],
    "Quantity": ["quantity", "quantity sold", "qty", "qty sold"],
    "Revenue": ["revenue", "sales", "total", "total revenue"],
}

OPTIONAL_COLUMNS_SYNONYMS = {
    "Waste": ["waste", "waste amount", "waste_amount"],
    "Cost": ["cost", "unit cost", "unit_cost"],
    "Supplier": ["supplier", "vendor"],
}


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_map = {c: normalize(c) for c in df.columns}
    rename_map = {}

    for standard, synonyms in REQUIRED_COLUMNS_SYNONYMS.items():
        candidates = [standard.lower(), *synonyms]
        match = next((col for col, low in lower_map.items() if low in candidates), None)
        if match is not None:
            rename_map[match] = standard

    for standard, synonyms in OPTIONAL_COLUMNS_SYNONYMS.items():
        candidates = [standard.lower(), *synonyms]
        match = next((col for col, low in lower_map.items() if low in candidates), None)
        if match is not None:
            rename_map[match] = standard

    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS_SYNONYMS.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        s = pd.to_datetime(df["Date"], errors="coerce")
        if s.isna().mean() > 0.3:
            s = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["Date"] = s
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = parse_dates(df)
    df = coerce_numeric(df, ["Price", "Quantity", "Revenue", "Waste", "Cost"])

    if "Revenue" not in df.columns or df["Revenue"].isna().all():
        df["Revenue"] = (df.get("Price", 0).fillna(0) * df.get("Quantity", 0).fillna(0))

    for col in ["Price", "Quantity", "Revenue", "Waste", "Cost"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Only require Product/Category; keep rows even if Date is NaT for non-time analyses
    drop_subset = [c for c in ["Product", "Category"] if c in df.columns]
    if drop_subset:
        df = df.dropna(subset=drop_subset).reset_index(drop=True)
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess_and_upload.py path\\to\\your.csv")
        sys.exit(2)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(2)

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded shape: {df.shape}")

    try:
        cleaned = preprocess(df)
    except Exception as e:
        print(f"Preprocess failed: {e}")
        sys.exit(1)

    export_dir = os.getenv(
        "PRICEOPTIMA_EXPORT_DIR",
        os.path.join(os.path.expanduser("~"), "Documents", "PriceOptima", "exports"),
    )
    os.makedirs(export_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(export_dir, f"preprocessed_{ts}.csv")
    cleaned.to_csv(out_csv, index=False)
    # Also save a features parquet in the same folder for notebook compatibility
    parquet_path = os.path.join(export_dir, "features_train.parquet")
    try:
        cleaned.to_parquet(parquet_path, index=False)
        print(f"Saved cleaned file: {out_csv}\nSaved parquet: {parquet_path}")
    except Exception as e:
        print(f"Saved cleaned file: {out_csv}\nParquet save skipped: {e}")

    # Upload original file to backend (backend also preprocesses and caches in-memory)
    base_url = os.getenv("PRICEOPTIMA_API", "http://127.0.0.1:8000")
    try:
        with open(input_path, "rb") as f:
            files = {"file": (os.path.basename(input_path), f, "text/csv")}
            r = requests.post(f"{base_url}/upload", files=files, timeout=60)
        if r.status_code != 200:
            print(f"Upload failed: HTTP {r.status_code} - {r.text}")
            sys.exit(1)
        data = r.json()
        print("Upload successful. Summary:")
        print({
            "records": data.get("summary", {}).get("totalRecords"),
            "products": data.get("summary", {}).get("products"),
            "categories": data.get("summary", {}).get("categories"),
        })
        # Optional: trigger EDA to warm the cache
        r2 = requests.post(f"{base_url}/eda", json={}, timeout=60)
        if r2.status_code == 200:
            print("EDA ready.")
        else:
            print(f"EDA call returned: HTTP {r2.status_code}")
    except Exception as e:
        print(f"Backend upload failed: {e}")
        sys.exit(1)

    print("Done. You can now click Start EDA Analysis in the UI.")


if __name__ == "__main__":
    main()


