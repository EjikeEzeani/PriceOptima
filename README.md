# PriceOptima - Dynamic Pricing Analytics Platform

A comprehensive AI-powered dynamic pricing analytics platform built with Next.js frontend and FastAPI backend.

## Features

- **Data Analysis**: Comprehensive EDA with statistical insights and trend analysis
- **AI Models**: Machine learning models for predictive pricing and optimization
- **RL Simulation**: Reinforcement learning for dynamic pricing strategies
- **Export Reports**: Generate comprehensive reports in multiple formats
- **Real-time Analytics**: Live monitoring and optimization recommendations
- **Performance Metrics**: Track revenue growth and optimization effectiveness

## Quick Start

### Frontend (Next.js)
```bash
npm install
npm run dev
```

### Backend (FastAPI)
```bash
pip install -r requirements.txt
python -m uvicorn api_backend:app --host 127.0.0.1 --port 8002 --reload
```

## Deployment

- **Frontend**: Deploy to Vercel
- **Backend**: Deploy to Render

## Project Structure

```
├── app/                    # Next.js frontend
├── src/                    # Python source code
├── data/                   # Data files
├── models/                 # Trained models
├── exports/                # Generated reports
└── figures/                # Visualizations
```

## API Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload data
- `GET /eda` - Exploratory data analysis
- `GET /ml` - Machine learning predictions
- `GET /rl` - Reinforcement learning simulation
- `GET /export` - Export reports

## License

MIT License
