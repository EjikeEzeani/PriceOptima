# Render Deployment Fix - Updated File Names

## Problem: Render Cannot Find Requirements Files
After renaming requirements files to prevent Vercel Python detection, Render can no longer find them.

## Solution: Updated Configuration

### For Super Minimal Backend:
- **Requirements File**: `backend_requirements_super_minimal.txt`
- **Build Command**: `pip install -r backend_requirements_super_minimal.txt`
- **Start Command**: `python -m uvicorn render_super_minimal:app --host 0.0.0.0 --port $PORT`

### For Bare Minimum Backend:
- **Requirements File**: `backend_requirements_bare_minimum.txt`
- **Build Command**: `pip install -r backend_requirements_bare_minimum.txt`
- **Start Command**: `python -m uvicorn render_bare_minimum:app --host 0.0.0.0 --port $PORT`

### For Ultra Light Backend:
- **Requirements File**: `backend_requirements_minimal.txt`
- **Build Command**: `pip install -r backend_requirements_minimal.txt`
- **Start Command**: `python -m uvicorn render_ultra_light:app --host 0.0.0.0 --port $PORT`

### For Render Optimized Backend:
- **Requirements File**: `backend_requirements_render.txt`
- **Build Command**: `pip install -r backend_requirements_render.txt`
- **Start Command**: `python -m uvicorn render_optimized_backend:app --host 0.0.0.0 --port $PORT`

## Environment Variables:
```
PYTHONUNBUFFERED=1
MALLOC_TRIM_THRESHOLD_=100000
MALLOC_MMAP_THRESHOLD_=131072
```

## Root Directory: (empty)

## This Fixes the Render Build Error!
