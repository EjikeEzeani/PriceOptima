# Dynamic Pricing for Waste Reduction - Reproducible Project

## Quick start (VS Code)

1. Create project folder and subfolders (data/raw, data/processed, notebooks, src, src/envs, models, figures, logs).
2. Copy files from this repository into project_root.
3. Create and activate venv:
   - `python -m venv venv`
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install requirements:
   - `pip install -r requirements.txt`
5. Generate merged dataset:
   - Option A (script): `python src/generate_simulated_pos.py --seed 42`
   - Option B (notebook): open `notebooks/01_data_merge.py` in VS Code and run all cells.
6. Preprocess & run pipeline:
   - Run notebooks in order:
     - `01_data_merge.py`
     - `02_preprocessing.py`
     - `03_eda.py`
     - `04_supervised_models.py`
     - `05_explainability.py`
     - `06_rl_environment.py`
     - `07_rl_training.py`
     - `08_evaluation.py`
   - Or run end-to-end: `python src/run_all.py`
7. Check `data/processed`, `models`, `figures`, `logs` for outputs.

## Notes
- Files are deterministic where possible (random seeds).
- If you have the real WFP CSV, place it in `data/raw/wfp_food_prices.csv`; otherwise the script will synthesize WFP-style data.
- See `validation.md` for expected metrics and troubleshooting tips.

