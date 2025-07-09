from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    file_path = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name


class ForecastModel(models.Model):
    MODEL_CHOICES = [
        ('prophet', 'Prophet'),
        ('arima', 'ARIMA'),
        ('lstm', 'LSTM'),
        ('ensemble', 'Ensemble'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_CHOICES)
    parameters = models.JSONField(default=dict)
    is_trained = models.BooleanField(default=False)
    accuracy_score = models.FloatField(null=True, blank=True)
    model_path = models.CharField(max_length=500, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.model_type}"


class ForecastResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    model = models.ForeignKey(ForecastModel, on_delete=models.CASCADE)
    forecast_horizon = models.IntegerField()
    confidence_interval = models.FloatField(default=0.95)
    predictions = models.JSONField()
    confidence_bounds = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Forecast {self.id} - {self.model.name}"


class ModelPerformance(models.Model):
    model = models.ForeignKey(ForecastModel, on_delete=models.CASCADE)
    metric_name = models.CharField(max_length=50)
    metric_value = models.FloatField()
    test_period_start = models.DateField()
    test_period_end = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.model.name} - {self.metric_name}: {self.metric_value}"