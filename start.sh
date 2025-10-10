#!/bin/bash
# Render startup script
cd /opt/render/project/src
python -m uvicorn render_optimized_backend:app --host 0.0.0.0 --port $PORT
