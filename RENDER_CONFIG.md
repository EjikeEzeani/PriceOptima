# Render Service Configuration

## Current Error:
```
Service Root Directory "/opt/render/project/src/render_optimized_backend.py" is missing.
```

## Solution: Update Render Service Settings

### Option 1: Use Root Directory (Simplest)
1. **Root Directory**: Leave empty or set to `/`
2. **Build Command**: `pip install -r backend_requirements_render.txt`
3. **Start Command**: `python -m uvicorn render_optimized_backend:app --host 0.0.0.0 --port $PORT`

### Option 2: Use Docker (Recommended)
1. **Enable Docker** in your Render service
2. **Use**: `Dockerfile.render`
3. **Deploy** - This handles everything automatically

### Option 3: Fix Directory Structure
1. **Root Directory**: `src`
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `python -m uvicorn render_optimized_backend:app --host 0.0.0.0 --port $PORT`

## Files Available:
- `render_optimized_backend.py` (in root)
- `backend_requirements_render.txt` (in root)
- `src/render_optimized_backend.py` (in src/)
- `src/requirements.txt` (in src/)
- `Dockerfile.render` (Docker option)

## Environment Variables to Add:
```
PYTHONUNBUFFERED=1
MALLOC_TRIM_THRESHOLD_=100000
MALLOC_MMAP_THRESHOLD_=131072
```
