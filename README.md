# Flight Demand Forecasting System

A production-ready Django web application for forecasting flight passenger demand using advanced machine learning models. Built specifically for Emirates Flight Catering operations.

## 🚀 Features

- **Multiple ML Models**: Prophet, ARIMA, LSTM, and Ensemble forecasting
- **Interactive Visualizations**: Beautiful charts with Plotly
- **Production Ready**: Proper error handling, caching, and logging
- **Responsive Design**: Modern Bootstrap 5 UI
- **RESTful API**: JSON endpoints for integration
- **Model Performance Tracking**: Comprehensive metrics and history

## 📊 Models Implemented

### 1. Prophet Model
- Facebook's robust time series forecasting tool
- Handles seasonality, trends, and holidays
- Best for data with strong seasonal patterns

### 2. ARIMA Model
- Classical statistical approach
- Auto-tuned parameters using AIC criterion
- Excellent for stationary time series

### 3. LSTM Neural Network
- Deep learning approach for complex patterns
- Learns long-term dependencies
- Handles non-linear relationships

### 4. Ensemble Model
- Combines all models with weighted averaging
- Maximizes prediction accuracy
- Reduces individual model bias

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip
- virtualenv (recommended)

### Setup Steps

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd flight_forecasting
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**
Create a `.env` file in the root directory:
```
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

5. **Database Setup**
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

6. **Collect Static Files**
```bash
python manage.py collectstatic
```

7. **Run the Application**
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to access the application.

## 📱 Usage

### Web Interface
1. Navigate to the dashboard
2. Select your preferred model
3. Set forecast horizon (1-60 months)
4. Choose confidence interval
5. Click "Generate Forecast"
6. View interactive charts and detailed results

### API Usage
```bash
# Generate forecast via API
curl -X POST http://127.0.0.1:8000/api/forecast/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "prophet",
    "forecast_horizon": 12,
    "confidence_interval": 0.95,
    "seasonality_mode": "multiplicative"
  }'