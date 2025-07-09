from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView
from django.core.cache import cache
import json
import logging
from .forms import ForecastParametersForm
from .services import ForecastingService
from .models import ForecastResult, ForecastModel

logger = logging.getLogger(__name__)


class DashboardView(TemplateView):
    template_name = 'forecasting/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = ForecastParametersForm()
        context['title'] = 'Flight Demand Forecasting System'
        return context


def generate_forecast_view(request):
    if request.method == 'POST':
        form = ForecastParametersForm(request.POST)
        if form.is_valid():
            try:
                # Get form data
                model_type = form.cleaned_data['model_type']
                forecast_horizon = form.cleaned_data['forecast_horizon']
                confidence_interval = form.cleaned_data['confidence_interval']
                seasonality_mode = form.cleaned_data['seasonality_mode']
                
                # Check cache first
                cache_key = f"forecast_{model_type}_{forecast_horizon}_{confidence_interval}_{seasonality_mode}"
                cached_result = cache.get(cache_key)
                
                if cached_result:
                    logger.info(f"Using cached forecast result for {cache_key}")
                    return render(request, 'forecasting/forecast_results.html', {
                        'results': cached_result,
                        'form': form
                    })
                
                # Generate forecast
                forecasting_service = ForecastingService()
                results = forecasting_service.generate_forecast(
                    model_type=model_type,
                    forecast_horizon=forecast_horizon,
                    confidence_interval=confidence_interval,
                    seasonality_mode=seasonality_mode
                )
                
                # Cache results for 1 hour
                cache.set(cache_key, results, 3600)
                
                # Save to database
                try:
                    forecast_model, created = ForecastModel.objects.get_or_create(
                        name=f"{model_type.upper()} Model",
                        model_type=model_type,
                        defaults={
                            'parameters': {
                                'forecast_horizon': forecast_horizon,
                                'confidence_interval': confidence_interval,
                                'seasonality_mode': seasonality_mode
                            }
                        }
                    )
                    
                    ForecastResult.objects.create(
                        user=request.user if request.user.is_authenticated else None,
                        model=forecast_model,
                        forecast_horizon=forecast_horizon,
                        confidence_interval=confidence_interval,
                        predictions=results['forecast_data'],
                        confidence_bounds={
                            'lower': results['forecast_data']['lower_bound'],
                            'upper': results['forecast_data']['upper_bound']
                        }
                    )
                except Exception as e:
                    logger.error(f"Error saving forecast result: {str(e)}")
                
                messages.success(request, f'Forecast generated successfully using {model_type.upper()} model!')
                return render(request, 'forecasting/forecast_results.html', {
                    'results': results,
                    'form': form
                })
                
            except Exception as e:
                logger.error(f"Error generating forecast: {str(e)}")
                messages.error(request, f'Error generating forecast: {str(e)}')
                return render(request, 'forecasting/dashboard.html', {
                    'form': form,
                    'title': 'Flight Demand Forecasting System'
                })
        else:
            messages.error(request, 'Please correct the errors in the form.')
    else:
        form = ForecastParametersForm()
    
    return render(request, 'forecasting/dashboard.html', {
        'form': form,
        'title': 'Flight Demand Forecasting System'
    })


@csrf_exempt
def api_forecast(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Validate required parameters
            model_type = data.get('model_type', 'prophet')
            forecast_horizon = int(data.get('forecast_horizon', 12))
            confidence_interval = float(data.get('confidence_interval', 0.95))
            seasonality_mode = data.get('seasonality_mode', 'multiplicative')
            
            # Generate forecast
            forecasting_service = ForecastingService()
            results = forecasting_service.generate_forecast(
                model_type=model_type,
                forecast_horizon=forecast_horizon,
                confidence_interval=confidence_interval,
                seasonality_mode=seasonality_mode
            )
            
            return JsonResponse({
                'success': True,
                'data': results
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def forecast_history_view(request):
    recent_forecasts = ForecastResult.objects.all()[:10]
    
    return render(request, 'forecasting/history.html', {
        'forecasts': recent_forecasts,
        'title': 'Forecast History'
    })