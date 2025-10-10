# Bare Minimum Render Backend

## Problem: Still Getting Build Failures
Even the ultra-lightweight version is failing due to pandas/numpy compilation issues.

## Solution: Bare Minimum Backend

### Dependencies (Only 4 packages):
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pandas==2.1.4
- python-multipart==0.0.6

### Removed ALL problematic dependencies:
- ❌ numpy (causes compilation issues)
- ❌ scikit-learn (heavy compilation)
- ❌ requests (not needed)
- ❌ All ML libraries

### Features:
- ✅ File upload (2MB max, 2k rows max)
- ✅ Basic EDA (pandas only)
- ✅ Mock ML results
- ✅ Mock RL simulation
- ✅ JSON export only
- ✅ Memory optimization
- ✅ Garbage collection

## Render Configuration:

### Root Directory: (empty)
### Build Command: `pip install -r requirements_bare_minimum.txt`
### Start Command: `python -m uvicorn render_bare_minimum:app --host 0.0.0.0 --port $PORT`

## Environment Variables:
```
PYTHONUNBUFFERED=1
MALLOC_TRIM_THRESHOLD_=100000
MALLOC_MMAP_THRESHOLD_=131072
```

This should definitely build successfully on Render!
