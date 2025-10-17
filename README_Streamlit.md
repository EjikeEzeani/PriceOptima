# PriceOptima Streamlit Dashboard

A comprehensive interactive dashboard for analyzing food price data, training machine learning models, and generating insights using Streamlit.

## 🚀 Features

### 📊 Dashboard Overview
- **Key Metrics**: Total records, unique items, states/markets, date range
- **Price Statistics**: Average, median, min/max prices with standard deviation
- **Quick Visualizations**: Price trends over time and top commodities by price
- **Model Status**: Display of trained models and their readiness

### 📈 Data Explorer
- **Interactive Data Preview**: Browse through your datasets with pagination
- **Basic Statistics**: Comprehensive statistical analysis of your data
- **Column Information**: Data types, null counts, and unique values
- **Interactive Filters**: Filter by commodities, states, markets, and date ranges
- **Advanced Visualizations**: 
  - Price distribution histograms
  - Time series analysis
  - Geographic analysis by state
  - Correlation matrices

### 🤖 Machine Learning
- **Model Selection**: Choose from pre-trained models
- **Model Information**: Display model parameters and characteristics
- **Feature Importance**: Visualize which features are most important
- **Model Predictions**: Compare predicted vs actual values
- **Performance Metrics**: R² score, MSE, and MAE calculations

### 📊 Insights & Analytics
- **Key Insights**: Price volatility analysis and trend identification
- **Seasonal Analysis**: Monthly price patterns and seasonal trends
- **Geographic Insights**: State-by-state price analysis with maps
- **Market Analysis**: Market-specific price comparisons

### 🔮 Forecasting
- **Interactive Forecasting**: Predict future prices with configurable parameters
- **Confidence Intervals**: Statistical confidence bounds for predictions
- **Trend Analysis**: Linear trend analysis with visualization
- **Forecast Summary**: Key metrics and detailed forecast tables

## 📁 Data Sources

The dashboard supports multiple data sources:

1. **Nigeria Food Prices (Detailed)**: Comprehensive dataset with geographic and commodity details
2. **Simple Food Prices**: Simplified dataset for quick analysis
3. **Rainfall Data**: Environmental data for correlation analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   python run_streamlit.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Dashboard**:
   Open your browser and navigate to `http://localhost:8501`

## 📊 Usage

### Getting Started
1. **Select Dataset**: Choose from available datasets in the sidebar
2. **Explore Data**: Use the Data Explorer to understand your data
3. **Train Models**: Use the ML page to work with pre-trained models
4. **Generate Insights**: Discover patterns and trends in your data
5. **Forecast Prices**: Predict future price movements

### Navigation
- Use the sidebar to switch between different pages
- Each page offers specific functionality for different analysis needs
- Interactive filters allow you to focus on specific data subsets

## 🔧 Configuration

### Data Paths
The dashboard automatically looks for data in the following locations:
- `DATASET/raw/wfp_food_prices_nga.csv` - Nigeria food prices
- `DATASET/raw/wfp_food_prices.csv` - Simple food prices
- `DATASET/raw/nga-rainfall-subnat-full.csv` - Rainfall data

### Model Loading
Pre-trained models are automatically loaded from the `models/` directory.

## 📈 Visualizations

The dashboard includes various interactive visualizations:

- **Line Charts**: Time series analysis and trend visualization
- **Bar Charts**: Comparative analysis and rankings
- **Histograms**: Distribution analysis
- **Scatter Plots**: Correlation and prediction analysis
- **Maps**: Geographic data visualization (when coordinates available)
- **Heatmaps**: Correlation matrices

## 🎯 Key Features

### Interactive Filtering
- Filter by date ranges, commodities, states, and markets
- Real-time updates of visualizations based on filters
- Preserve filter state across page navigation

### Machine Learning Integration
- Load and analyze pre-trained models
- Feature importance visualization
- Model performance evaluation
- Prediction vs actual comparison

### Forecasting Capabilities
- Configurable forecast periods (7-90 days)
- Confidence interval visualization
- Trend analysis and price change predictions
- Multiple commodity support

## 🔍 Data Requirements

### Required Columns
- `date`: Date column for time series analysis
- `price`: Price data for analysis
- `commodity`: Product/commodity information

### Optional Columns
- `admin1`: State/region information
- `market`: Market location
- `latitude`/`longitude`: Geographic coordinates
- `category`: Product category

## 🚨 Troubleshooting

### Common Issues

1. **Data Not Loading**:
   - Check file paths in the `DATASET/raw/` directory
   - Ensure CSV files have proper headers
   - Verify data format matches expected structure

2. **Models Not Found**:
   - Ensure models are in the `models/` directory
   - Check model files are in `.joblib` format
   - Verify model compatibility

3. **Visualization Errors**:
   - Check data has required columns
   - Ensure numeric columns contain valid data
   - Verify date columns are properly formatted

### Performance Tips

- Use filters to reduce data size for better performance
- Limit forecast periods for faster processing
- Close unused browser tabs to free memory

## 📝 Notes

- The dashboard is designed for interactive exploration
- All visualizations are responsive and mobile-friendly
- Data is cached for improved performance
- Models are loaded once and reused across sessions

## 🤝 Contributing

To contribute to the dashboard:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the PriceOptima system. Please refer to the main project license for usage terms.

