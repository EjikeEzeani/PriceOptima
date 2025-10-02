# Local API Configuration

To ensure the frontend always talks to your local backend:

1) Start backend
```
python -m uvicorn api_backend:app --host 127.0.0.1 --port 8000 --reload
```

2) Start frontend (same shell sets env var)
```
Set-Location "C:\Users\USER\Downloads\Msc Project\dynamic-pricing-dashboard"
$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npx next dev -p 3001
```

Open http://localhost:3001, upload CSV, then Start EDA Analysis.


