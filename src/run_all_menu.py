"""
src/run_all_menu.py
Interactive CLI to run pipeline stages manually.
"""

import os
import yaml
from src import pipeline

# ------------------------------
# Load config
# ------------------------------
CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"✅ Loaded config from {CONFIG_PATH}")
else:
    print("⚠️ config.yaml not found. Using defaults.")
    cfg = {}

# ------------------------------
# Menu Loop
# ------------------------------
menu = """
📌 Select Stage to Run:
1. Merge Data
2. Preprocess Data
3. Train ML / RL
4. Evaluate Models
5. Launch Dashboard (Streamlit)
6. Run All
0. Exit
👉 Enter choice: """

while True:
    choice = input(menu).strip()
    if choice == "1":
        pipeline.merge_data()
    elif choice == "2":
        pipeline.preprocess_data()
    elif choice == "3":
        pipeline.train_models()
    elif choice == "4":
        pipeline.evaluate_models()
    elif choice == "5":
        os.system("streamlit run dashboard.py")
    elif choice == "6":
        pipeline.merge_data()
        pipeline.preprocess_data()
        pipeline.train_models()
        pipeline.evaluate_models()
        os.system("streamlit run dashboard.py")
    elif choice == "0":
        print("👋 Exiting...")
        break
    else:
        print("❌ Invalid choice, try again.")

