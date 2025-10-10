# Ultra Lightweight Render Backend

## Problem: Build Failed
The previous backend failed to build on Render due to heavy ML dependencies (scikit-learn, numpy compilation issues).

## Solution: Ultra Lightweight Backend

### Files:
- `render_ultra_light.py` - No heavy ML dependencies
- `requirements_minimal.txt` - Only essential packages

### Dependencies Removed:
- ❌ scikit-learn (causes compilation issues)
- ❌ xgboost (heavy compilation)
- ❌ shap (heavy dependencies)
- ❌ matplotlib (GUI dependencies)
- ❌ seaborn (heavy plotting)

### Dependencies Kept:
- ✅ fastapi (web framework)
- ✅ uvicorn (ASGI server)
- ✅ pandas (data processing)
- ✅ numpy (basic math)
- ✅ python-multipart (file uploads)
- ✅ requests (HTTP client)

### Features:
- ✅ File upload (5MB max, 5k rows max)
- ✅ Basic EDA (no heavy computation)
- ✅ Mock ML results (no actual training)
- ✅ Mock RL simulation
- ✅ JSON export only
- ✅ Memory optimization
- ✅ Garbage collection

## Render Configuration:

### Root Directory: (empty)
### Build Command: `pip install -r requirements_minimal.txt`
### Start Command: `python -m uvicorn render_ultra_light:app --host 0.0.0.0 --port $PORT`

## Environment Variables:
```
PYTHONUNBUFFERED=1
MALLOC_TRIM_THRESHOLD_=100000
MALLOC_MMAP_THRESHOLD_=131072
```

This should build successfully on Render!
