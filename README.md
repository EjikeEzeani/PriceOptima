# PriceOptima - AI-Powered Dynamic Pricing Analytics Platform

A comprehensive AI-powered dynamic pricing analytics platform built with Next.js frontend and FastAPI backend.

## Features

- **Data Analysis**: Comprehensive EDA with statistical insights and trend analysis
- **AI Models**: Machine learning models for predictive pricing and optimization
- **RL Simulation**: Reinforcement learning for dynamic pricing strategies
- **Export Reports**: Generate comprehensive reports in multiple formats
- **Real-time Analytics**: Live monitoring and optimization recommendations
- **Performance Metrics**: Track revenue growth and optimization effectiveness

## Quick Start

### Prerequisites
- Node.js 22.x
- Python 3.11+
- npm 10.x

### Local Development

#### 1. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python -m uvicorn backend.api_backend:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at `http://127.0.0.1:8000`

#### 2. Frontend Setup
```bash
# Install dependencies
npm install

# Create environment file
cp env.local.example .env.local

# Start the frontend server
npm run dev
```

The frontend will be available at `http://localhost:3000`

#### 3. Environment Configuration
Create `.env.local` in the root directory:
```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

**Important**: Restart the Next.js server after changing `.env.local`:
```bash
npm run dev
```

### Testing

#### Backend Tests
```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Run tests
pytest tests/backend/test_upload.py -v
```

#### Integration Tests
```bash
# Make script executable
chmod +x tests/integration/upload_smoke.sh

# Run integration test
API_BASE_URL=http://127.0.0.1:8000 ./tests/integration/upload_smoke.sh
```

## Deployment

### Backend (Render)

1. **Create a new Web Service on Render**
2. **Connect your GitHub repository**
3. **Configure settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn render_super_minimal:app --host 0.0.0.0 --port $PORT`
   - **Python Version**: `3.11`
4. **Deploy**

### Frontend (Vercel)

1. **Import project to Vercel**
2. **Configure environment variables**:
   - `NEXT_PUBLIC_API_BASE_URL`: `https://your-render-app.onrender.com`
3. **Deploy**

### Post-Deployment Testing

1. **Check backend health**: `https://your-render-app.onrender.com/health`
2. **Test file upload** in deployed frontend
3. **Verify no CORS errors** in browser console

## Troubleshooting

### Common Issues

#### "Failed to fetch" Error
- ✅ Check backend is running: `curl http://127.0.0.1:8000/health`
- ✅ Verify environment variable: `NEXT_PUBLIC_API_BASE_URL`
- ✅ Check CORS configuration in backend
- ✅ Restart Next.js after changing `.env.local`

#### Upload Errors
- ✅ Ensure CSV has required columns: Date, Product Name, Category, Price, Quantity Sold, Revenue
- ✅ Check file size (max 50MB)
- ✅ Verify file is valid CSV format

#### CORS Issues
- ✅ Backend allows all origins by default (`PRICEOPTIMA_ALLOW_ALL_ORIGINS=1`)
- ✅ For production, set specific origins in `PRICEOPTIMA_ALLOWED_ORIGINS`

### Debugging

#### Backend Logs
```bash
# Check Render logs for errors
# Look for upload processing logs
```

#### Frontend Console
```bash
# Open browser DevTools
# Check Network tab for failed requests
# Look for detailed error messages in console
```

## Project Structure

```
├── backend/
│   ├── api_backend.py          # Main FastAPI application
│   ├── requirements.txt        # Python dependencies
│   └── render_super_minimal.py # Render entry point
├── components/                 # React components
├── lib/
│   └── api.ts                 # API client with retry logic
├── tests/
│   ├── backend/               # Backend unit tests
│   └── integration/          # Integration tests
├── package.json               # Frontend dependencies
├── vercel.json               # Vercel configuration
├── requirements.txt          # Root requirements
└── README.md                # This file
```

## API Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload CSV data (max 50MB)
- `POST /eda` - Exploratory data analysis
- `POST /ml` - Machine learning predictions
- `POST /rl` - Reinforcement learning simulation
- `POST /export` - Export reports

## Data Requirements

### Required Columns
- Date (or Date_Time, DateTime)
- Product Name (or Product, Product_Name)
- Category (or Cat, Product_Category)
- Price (or Unit_Price, Unit Price)
- Quantity Sold (or Quantity, Quantity_Sold, Qty, Qty Sold)
- Revenue (or Total_Revenue, Total Revenue, Sales)

### Optional Columns
- Waste Amount
- Cost
- Supplier

### File Format
- CSV format preferred
- Maximum 50MB per file
- Date format: YYYY-MM-DD, DD/MM/YYYY, or MM/DD/YYYY

## License

MIT License