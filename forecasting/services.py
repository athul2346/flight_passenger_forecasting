import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from django.conf import settings
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    # def load_data(self):
    #     # Generate enhanced airline passenger dataset
    #     dates = pd.date_range(start='1949-01-01', end='2025-06-01', freq='MS')
        
    #     # Base pattern from classic dataset with realistic growth
    #     base_values = []
    #     for i, date in enumerate(dates):
    #         year = date.year
    #         month = date.month
            
    #         # Long-term trend
    #         trend = 100 + (year - 1949) * 15
            
    #         # Seasonal pattern
    #         seasonal = 20 * np.sin(2 * np.pi * (month - 1) / 12) + 10 * np.sin(4 * np.pi * (month - 1) / 12)
            
    #         # Growth acceleration after 1970
    #         if year > 1970:
    #             trend *= 1.5
    #         if year > 1990:
    #             trend *= 1.2
    #         if year > 2010:
    #             trend *= 1.1
                
    #         # COVID impact (2020-2022)
    #         covid_impact = 1.0
    #         if 2020 <= year <= 2022:
    #             covid_impact = 0.3 + 0.35 * ((year - 2020) / 2)
                
    #         # Add realistic noise
    #         noise = np.random.normal(0, 10)
            
    #         value = (trend + seasonal + noise) * covid_impact
    #         base_values.append(max(50, value))  # Minimum value
            
    #     df = pd.DataFrame({
    #         'date': dates,
    #         'passengers': base_values
    #     })
        
    #     return df
    def load_data(self):
        # Generate dates (1949-01-01 to 2025-06-01)
        dates = pd.date_range(start='1949-01-01', end='2025-06-01', freq='MS')
        num_months = len(dates)
        
        # Initialize DataFrame with date column
        df = pd.DataFrame({'date': dates})
        
        # --- 1. Passengers (Target Variable) ---
        # (Your existing logic with slight optimization)
        df['passengers'] = [
            max(50, (
                (100 + (date.year - 1949) * 15 * 
                (1.5 if date.year > 1970 else 1) * 
                (1.2 if date.year > 1990 else 1) * 
                (1.1 if date.year > 2010 else 1)) + 
                (20 * np.sin(2 * np.pi * (date.month - 1) / 12) + 
                10 * np.sin(4 * np.pi * (date.month - 1) / 12)) + 
                np.random.normal(0, 10)
            ) * (0.3 + 0.35 * ((date.year - 2020) / 2) if 2020 <= date.year <= 2022 else 1.0))
            for date in dates
        ]
        
        # --- 2. Synthetic Fuel Prices ---
        # Base price with inflation trend + seasonal variation + random shocks
        df['fuel_price'] = (
            1.5 + 
            (dates.year - 1949) * 0.03 +  # Inflation trend
            0.2 * np.sin(2 * np.pi * (dates.month - 1) / 12) +  # Seasonal variation
            np.random.normal(0, 0.1, num_months))  # Random shocks
        
        # --- 3. Holiday Flags ---
        # Peaks in Jan (New Year), Jul (Summer), Dec (Christmas)
        df['is_holiday'] = [1 if m in [1, 7, 12] else 0 for m in dates.month]
        
        # --- 4. Economic Indicator (e.g., GDP Growth) ---
        df['gdp_growth'] = (
            2.5 + 
            0.5 * np.sin(2 * np.pi * (dates.year - 1949) / 10) +  # Economic cycles
            np.random.normal(0, 0.3, num_months))
        
        # --- 5. Competitor Prices ---
        df['competitor_price'] = df['fuel_price'] * (
            1.1 +  # 10% premium
            0.05 * np.random.randn(num_months))  # Random variation
        
        return df
    
    def prepare_data(self, df):
        # Ensure proper datetime format
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Split data
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]
        
        return train_data, test_data
    
    def create_sequences(self, data, seq_length=12):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)


class ProphetForecaster:
    def __init__(self):
        self.model = None
        self.trained = False
        
    def train(self, train_data, seasonality_mode='multiplicative'):
        # Prepare data for Prophet
        prophet_data = train_data.rename(columns={'date': 'ds', 'passengers': 'y'})
        
        # Initialize and train model
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        self.model.fit(prophet_data)
        self.trained = True
        
    def predict(self, periods, confidence_interval=0.95):
        if not self.trained:
            raise ValueError("Model must be trained first")
            
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='MS')
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Extract predictions and confidence intervals
        predictions = forecast['yhat'].tail(periods).values
        lower_bound = forecast['yhat_lower'].tail(periods).values
        upper_bound = forecast['yhat_upper'].tail(periods).values
        
        return predictions, lower_bound, upper_bound


class ARIMAForecaster:
    def __init__(self):
        self.model = None
        self.trained = False
        self.order = (2, 1, 2)  # Default order
        
    def find_best_order(self, data, max_p=3, max_d=2, max_q=3):
        best_aic = np.inf
        best_order = (0, 0, 0)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
                        
        return best_order
        
    def train(self, train_data):
        data = train_data['passengers'].values
        
        # Find best order
        self.order = self.find_best_order(data)
        
        # Train model
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        self.trained = True
        
    def predict(self, periods, confidence_interval=0.95):
        if not self.trained:
            raise ValueError("Model must be trained first")
            
        # Generate forecast
        forecast = self.fitted_model.forecast(steps=periods)
        conf_int = self.fitted_model.get_forecast(steps=periods).conf_int(alpha=1-confidence_interval)
        
        predictions = forecast.values
        lower_bound = conf_int.iloc[:, 0].values
        upper_bound = conf_int.iloc[:, 1].values
        
        return predictions, lower_bound, upper_bound


class LSTMForecaster:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.trained = False
        self.seq_length = 12
        
    def train(self, train_data):
        data = train_data['passengers'].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        processor = DataProcessor()
        X, y = processor.create_sequences(scaled_data.flatten(), self.seq_length)
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.seq_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        self.model.fit(X, y, batch_size=32, epochs=50, verbose=0)
        self.trained = True
        
    def predict(self, periods, test_data):
        if not self.trained:
            raise ValueError("Model must be trained first")
            
        # Get last sequence from test data
        last_sequence = test_data['passengers'].tail(self.seq_length).values
        last_sequence = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            # Reshape for prediction
            X = current_sequence.reshape((1, self.seq_length, 1))
            
            # Predict next value
            next_pred = self.model.predict(X, verbose=0)[0][0]
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
            
        # Scale back predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Simple confidence intervals (±10% of prediction)
        lower_bound = predictions * 0.9
        upper_bound = predictions * 1.1
        
        return predictions, lower_bound, upper_bound


class EnsembleForecaster:
    def __init__(self):
        self.prophet = ProphetForecaster()
        self.arima = ARIMAForecaster()
        self.lstm = LSTMForecaster()
        self.weights = [0.5, 0.3, 0.2]  # Prophet, ARIMA, LSTM
        
    def train(self, train_data, seasonality_mode='multiplicative'):
        self.prophet.train(train_data, seasonality_mode)
        self.arima.train(train_data)
        self.lstm.train(train_data)
        
    def predict(self, periods, confidence_interval=0.95, test_data=None):
        # Get predictions from all models
        prophet_pred, prophet_lower, prophet_upper = self.prophet.predict(periods, confidence_interval)
        arima_pred, arima_lower, arima_upper = self.arima.predict(periods, confidence_interval)
        lstm_pred, lstm_lower, lstm_upper = self.lstm.predict(periods, test_data)
        
        # Weighted ensemble
        predictions = (self.weights[0] * prophet_pred + 
                      self.weights[1] * arima_pred + 
                      self.weights[2] * lstm_pred)
        
        lower_bound = (self.weights[0] * prophet_lower + 
                      self.weights[1] * arima_lower + 
                      self.weights[2] * lstm_lower)
        
        upper_bound = (self.weights[0] * prophet_upper + 
                      self.weights[1] * arima_upper + 
                      self.weights[2] * lstm_upper)
        
        return predictions, lower_bound, upper_bound


class ModelEvaluator:
    @staticmethod
    def evaluate_model(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }


class ForecastingService:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.evaluator = ModelEvaluator()
        
    def generate_forecast(self, model_type, forecast_horizon, confidence_interval=0.95, seasonality_mode='multiplicative'):
        # Load and prepare data
        data = self.data_processor.load_data()
        train_data, test_data = self.data_processor.prepare_data(data)
        
        # Select and train model
        if model_type == 'prophet':
            forecaster = ProphetForecaster()
            forecaster.train(train_data, seasonality_mode)
            predictions, lower_bound, upper_bound = forecaster.predict(forecast_horizon, confidence_interval)
        elif model_type == 'arima':
            forecaster = ARIMAForecaster()
            forecaster.train(train_data)
            predictions, lower_bound, upper_bound = forecaster.predict(forecast_horizon, confidence_interval)
        elif model_type == 'lstm':
            forecaster = LSTMForecaster()
            forecaster.train(train_data)
            predictions, lower_bound, upper_bound = forecaster.predict(forecast_horizon, test_data)
        elif model_type == 'ensemble':
            forecaster = EnsembleForecaster()
            forecaster.train(train_data, seasonality_mode)
            predictions, lower_bound, upper_bound = forecaster.predict(forecast_horizon, confidence_interval, test_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create future dates
        last_date = data['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=32), periods=forecast_horizon, freq='MS')
        
        # Prepare results
        results = {
            'historical_data': {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'values': data['passengers'].tolist()
            },
            'forecast_data': {
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'predictions': predictions.tolist(),
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist()
            },
            'model_info': {
                'type': model_type,
                'forecast_horizon': forecast_horizon,
                'confidence_interval': confidence_interval
            }
        }
        
        return results