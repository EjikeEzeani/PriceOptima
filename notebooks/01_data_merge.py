#!/usr/bin/env python3
"""
Safe replacement for notebooks/01_data_merge.py

- Uses robust path resolution relative to project root
- Avoids importing/installed packages via setup.py
- Attempts to import src.generate_simulated_pos; if unavailable,
  falls back to executing the script file directly with runpy
- Prints clear error messages for missing files or dependencies
"""

from pathlib import Path
import sys
import traceback
import yaml
import pandas as pd

# === Utility helpers ===
def ensure_pyarrow_or_fastparquet():
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            return None

def abs_project_root():
    # assume this script lives in <project>/notebooks/
    return Path(__file__).resolve().parents[1]

# === Setup paths ===
PROJECT_ROOT = abs_project_root()
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

print(f"Project root detected at: {PROJECT_ROOT}")
print(f"Looking for config at: {CONFIG_PATH}")

# === Load config safely ===
if not CONFIG_PATH.exists():
    print("ERROR: config.yaml not found at expected location.")
    print("Make sure you run this script from the project root or place config.yaml at the project root.")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print("Config loaded.")

# === Ensure parquet reader backend ===
backend = ensure_pyarrow_or_fastparquet()
if backend is None:
    print("\nERROR: No parquet engine found. Install one of:")
    print("    pip install pyarrow")
    print("or  pip install fastparquet")
    sys.exit(1)
else:
    print(f"Using parquet engine: {backend}")

# === Safely run the generator ===
# The generator is expected at src/generate_simulated_pos.py with a main(...) function
GEN_MODULE = "src.generate_simulated_pos"
gen_called = False

# Approach:
# 1) Try normal import if src is a proper package
# 2) If that fails, attempt to run the file directly with runpy from src/generate_simulated_pos.py
try:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))  # allow normal imports like `import src...`
    print("\nAttempting to import", GEN_MODULE)
    gen_mod = __import__(GEN_MODULE, fromlist=["main", "set_seeds"])
    main_func = getattr(gen_mod, "main", None)
    set_seeds = getattr(gen_mod, "set_seeds", None)
    if main_func is None:
        raise ImportError(f"{GEN_MODULE} does not expose a `main` function.")
    # call set_seeds if available and cfg has simulation.seed
    seed = cfg.get("simulation", {}).get("seed", None)
    if set_seeds and seed is not None:
        try:
            set_seeds(seed)
            print("Set random seeds via set_seeds().")
        except Exception:
            print("Warning: set_seeds() exists but failed; continuing.")
    # call main
    print("Calling main(...) from imported module.")
    main_func_args = []
    try:
        # prefer calling with argparse.Namespace if signature expects it
        import argparse
        main_func(argparse.Namespace(seed=seed) if seed is not None else argparse.Namespace())
    except TypeError:
        # fallback: call with no args
        main_func()
    gen_called = True
except Exception as e_import:
    print("Importing module failed:", str(e_import))
    print("Falling back to executing the script file directly (runpy).")
    try:
        import runpy
        gen_file = SRC_DIR / "generate_simulated_pos.py"
        if not gen_file.exists():
            raise FileNotFoundError(f"{gen_file} not found.")
        # run the file in its own namespace
        runpy.run_path(str(gen_file), run_name="__main__")
        gen_called = True
    except Exception as e_run:
        print("Failed to execute the generator script.")
        traceback.print_exc()
        sys.exit(1)

if gen_called:
    print("Data generation step completed (or at least invoked).")

# === Load merged file ===
proc_dir = Path(cfg["data"]["processed_dir"])
merged_fname = cfg["data"]["merged_filename"]
merged_path = (PROJECT_ROOT / proc_dir / merged_fname) if not (Path(cfg["data"]["processed_dir"]).is_absolute()) else Path(cfg["data"]["processed_dir"]) / merged_fname

print(f"\nLooking for merged file at: {merged_path}")

if not merged_path.exists():
    print("ERROR: merged input parquet file not found.")
    print("List files in processed dir:", list((PROJECT_ROOT / proc_dir).glob("*")) if (PROJECT_ROOT / proc_dir).exists() else "(processed dir not found)")
    sys.exit(1)

try:
    # read parquet
    merged = pd.read_parquet(merged_path)
    print("Merged shape:", merged.shape)
    # basic checks
    print("Columns:", list(merged.columns))
    # if 'revenue' expected, dropna safely
    if "revenue" in merged.columns:
        merged = merged.dropna(subset=["revenue"])
        print("Dropped rows with missing revenue. New shape:", merged.shape)
    else:
        print("Warning: 'revenue' column not present in merged dataset.")
    # show first rows (safe print)
    print("\nFirst 5 rows:")
    print(merged.head(5).to_string(index=False))
except Exception:
    print("Failed to read the parquet file. Traceback:")
    traceback.print_exc()
    sys.exit(1)

print("\nScript finished successfully.")


