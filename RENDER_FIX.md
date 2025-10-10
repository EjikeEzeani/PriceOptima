# Render Configuration Guide

## Fix for "Service Root Directory" Error

The error occurs because Render is looking for the file in the wrong location.

### Solution 1: Update Render Service Settings

1. **Go to your Render service settings**
2. **Update "Root Directory" to:** `src`
3. **Update "Build Command" to:** `pip install -r requirements.txt`
4. **Update "Start Command" to:** `python -m uvicorn render_optimized_backend:app --host 0.0.0.0 --port $PORT`

### Solution 2: Use Docker (Recommended)

1. **Enable Docker** in your Render service
2. **Use Dockerfile.render** (already created)
3. **Deploy** - This will handle the directory structure automatically

### Solution 3: Move Files to Root

If the above doesn't work, move the files to the project root:
- Move `render_optimized_backend.py` to root
- Move `render_requirements.txt` to root as `requirements.txt`
- Set Root Directory to: `/` (default)

## Current File Structure

```
PriceOptima/
├── src/
│   ├── render_optimized_backend.py
│   └── requirements.txt
├── render_optimized_backend.py (backup)
├── render_requirements.txt (backup)
├── Dockerfile.render
└── start.sh
```

## Environment Variables

Add these to your Render service:
```
PYTHONUNBUFFERED=1
MALLOC_TRIM_THRESHOLD_=100000
MALLOC_MMAP_THRESHOLD_=131072
```
