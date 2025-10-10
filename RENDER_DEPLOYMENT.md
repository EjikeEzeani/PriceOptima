# Render Deployment Configuration for Memory-Optimized Backend

## Files to Upload to Render

1. **render_optimized_backend.py** - Main application file
2. **render_requirements.txt** - Python dependencies
3. **Dockerfile.render** - Docker configuration (optional)

## Render Service Configuration

### Environment Variables
Add these to your Render service settings:

```
PYTHONUNBUFFERED=1
MALLOC_TRIM_THRESHOLD_=100000
MALLOC_MMAP_THRESHOLD_=131072
```

### Build Command
```
pip install -r render_requirements.txt
```

### Start Command
```
python -m uvicorn render_optimized_backend:app --host 0.0.0.0 --port $PORT
```

### Instance Type
- **Minimum**: Starter ($7/month) - 512MB RAM
- **Recommended**: Standard ($25/month) - 1GB RAM
- **For heavy usage**: Pro ($85/month) - 2GB RAM

## Key Optimizations

1. **Memory Limits**: Max 10k rows, 10MB files
2. **Data Types**: Optimized pandas dtypes
3. **Garbage Collection**: Automatic memory cleanup
4. **Lightweight ML**: Simple linear regression only
5. **Efficient EDA**: Basic statistics only
6. **JSON Export**: No heavy file formats

## Testing Locally

```bash
# Test the optimized backend
python -c "from render_optimized_backend import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"
```

## Deployment Steps

1. Upload `render_optimized_backend.py` to your Render service
2. Update requirements.txt with `render_requirements.txt` content
3. Set environment variables
4. Deploy and monitor memory usage
5. Update frontend to point to new backend URL
