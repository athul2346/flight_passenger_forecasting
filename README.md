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

git clone <your-repository-url>
cd flight_forecasting


2. **Create virtual environment**

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. **Install dependencies**

pip install -r requirements.txt


4. **Environment Configuration**
Create a `.env` file in the root directory:

SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1


5. **Database Setup**

python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser


6. **Collect Static Files**

python manage.py collectstatic

7. **Run the Application**
python manage.py runserver

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

curl -X POST http://127.0.0.1:8000/api/forecast/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "prophet",
    "forecast_horizon": 12,
    "confidence_interval": 0.95,
    "seasonality_mode": "multiplicative"
  }'


# Key Points

1. The project adopts a multi-model approach because the optimal forecasting model depends heavily on the dataset and the specific task at hand. Rather than relying on a single model, I have implemented multiple options—including ARIMA for linear trends (which is simpler than Prophet or LSTM) and an ensemble method to combine their strengths—to ensure flexibility and improved accuracy. While the dataset used in this project performs well with Prophet, I included alternative models to accommodate different data patterns. The ensemble approach further enhances predictions by leveraging the complementary strengths of individual models.
2. I chose air passenger data because it’s the closest public proxy for flight meal demand. Because more passenger means more meals.Even thoough the data generated in this project is synthetic It has been created to duplicate real world data with world scenarios taken into consideration.
3. Please note preprocessing techniques like dropping the null coloumns is not being done here as it is a synthetically generated data because I wanted to give emphasise to a multi model approach.Preprocessing needed for each model is being done for each model at their code level.
4. The model was trained within two days with limited data so there may be accuracy issues. The data was generated because we needed a bulk data which to ensure max accuracy which was not availabale in the internet.Also the prediction may be different from real world situation
5. I have tried creating models with kaggle datasets also but thought this was a better approach as this is something related to business of the company. Also most of the datasets available in the internet are also dummy datasets with values being inserted which is the same being done in dummy data generation.
