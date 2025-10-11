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

### Frontend (Next.js)
```bash
cd dynamic-pricing-dashboard
npm install
npm run dev
```

### Backend (FastAPI)
```bash
pip install -r requirements.txt
python api_backend.py
```

## Deployment

- **Frontend**: Deploy to Vercel
- **Backend**: Deploy to Render

## Project Structure

```
├── dynamic-pricing-dashboard/    # Next.js frontend
├── api_backend.py               # FastAPI backend
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## API Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload data
- `POST /eda` - Exploratory data analysis
- `POST /ml` - Machine learning predictions
- `POST /rl` - Reinforcement learning simulation
- `POST /export` - Export reports

## License

MIT License