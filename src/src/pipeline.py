"""
Pipeline Functions: merge, preprocess, train, evaluate
This is imported by app.py (FastAPI) and run_all_menu.py (CLI).
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "merged_input_dataset.csv"

def merge_data():
    print("🔄 Merging raw datasets...")
    # Replace with your actual merging logic
    df = pd.DataFrame({"product": ["Tomatoes", "Rice"], "price": [100, 200]})
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"✅ Data merged → {DATA_PROCESSED}")
    return df

def preprocess_data():
    print("🔄 Preprocessing dataset...")
    if DATA_PROCESSED.exists():
        df = pd.read_csv(DATA_PROCESSED)
        # Replace with actual preprocessing
        df["price_normalized"] = df["price"] / df["price"].max()
        df.to_csv(DATA_PROCESSED, index=False)
        print("✅ Preprocessing complete")
        return df
    else:
        raise FileNotFoundError("Merged dataset not found. Run merge first.")

def train_models():
    print("🔄 Training ML / RL models...")
    # Replace with actual ML training logic
    print("✅ Models trained")

def evaluate_models():
    print("🔄 Evaluating models...")
    # Replace with actual evaluation logic
    print("✅ Evaluation complete")
