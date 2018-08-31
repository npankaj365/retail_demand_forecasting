# from pyramid.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

loaded_model = ARIMAResults.load('arima_model.pkl')
forecast = loaded_model.forecast(steps=10)[0]   
print(forecast)

# print(loaded_model.aic())