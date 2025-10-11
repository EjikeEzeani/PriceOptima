FROM python:3.13-slim

WORKDIR /app
COPY . /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Auto-run new app.py
CMD ["python", "app.py"]
