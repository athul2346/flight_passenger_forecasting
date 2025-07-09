from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column, HTML
from .models import ForecastModel


class ForecastParametersForm(forms.Form):
    MODEL_CHOICES = [
        ('prophet', 'Prophet (Recommended)'),
        ('arima', 'ARIMA'),
        ('lstm', 'LSTM Neural Network'),
        ('ensemble', 'Ensemble (All Models)'),
    ]
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        initial='prophet',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    forecast_horizon = forms.IntegerField(
        min_value=1,
        max_value=60,
        initial=12,
        help_text="Number of months to forecast (1-60)",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    confidence_interval = forms.FloatField(
        min_value=0.8,
        max_value=0.99,
        initial=0.95,
        help_text="Confidence interval (0.8-0.99)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    
    seasonality_mode = forms.ChoiceField(
        choices=[('additive', 'Additive'), ('multiplicative', 'Multiplicative')],
        initial='multiplicative',
        help_text="Seasonality mode for Prophet model",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.layout = Layout(
            Row(
                Column('model_type', css_class='form-group col-md-6 mb-3'),
                Column('forecast_horizon', css_class='form-group col-md-6 mb-3'),
            ),
            Row(
                Column('confidence_interval', css_class='form-group col-md-6 mb-3'),
                Column('seasonality_mode', css_class='form-group col-md-6 mb-3'),
            ),
            HTML('<hr>'),
            Submit('submit', 'Generate Forecast', css_class='btn btn-primary btn-lg')
        )