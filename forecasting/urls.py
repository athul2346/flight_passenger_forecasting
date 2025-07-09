from django.urls import path
from . import views

app_name = 'forecasting'

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('generate/', views.generate_forecast_view, name='generate_forecast'),
    path('api/forecast/', views.api_forecast, name='api_forecast'),
    path('history/', views.forecast_history_view, name='history'),
]