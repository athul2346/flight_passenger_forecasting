from django.contrib import admin
from .models import Dataset, ForecastModel, ForecastResult, ModelPerformance


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'created_at', 'updated_at')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(ForecastModel)
class ForecastModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'is_trained', 'accuracy_score', 'created_at')
    list_filter = ('model_type', 'is_trained')
    search_fields = ('name',)
    readonly_fields = ('created_at',)


@admin.register(ForecastResult)
class ForecastResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'model', 'user', 'forecast_horizon', 'confidence_interval', 'created_at')
    list_filter = ('model__model_type', 'created_at')
    search_fields = ('user__username', 'model__name')
    readonly_fields = ('created_at',)


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ('model', 'metric_name', 'metric_value', 'test_period_start', 'test_period_end')
    list_filter = ('metric_name', 'model__model_type')
    search_fields = ('model__name',)