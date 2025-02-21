from statsmodels.tsa.arima.model import ARIMA
import pandas as pd 


df = pd.read_csv('Outputs/output_monthly.csv')
df_actual=pd.read_csv("bonds_10yr_data.csv")

actual_series = df_actual['AAA_Bond_Yield']
predicted_series = df['AAA']

look_back = 12
forecast_steps = 6

if len(actual_series) < look_back or len(predicted_series) < look_back:
    raise ValueError("Not enough data points.")

actual_train = actual_series[-look_back:]
predicted_train = predicted_series[-look_back:]

model_actual = ARIMA(actual_train, order=(1, 0, 0))
model_actual_fit = model_actual.fit()

model_predicted = ARIMA(predicted_train, order=(1, 0, 0))
model_predicted_fit = model_predicted.fit()

actual_forecast = model_actual_fit.forecast(steps=forecast_steps)
predicted_forecast = model_predicted_fit.forecast(steps=forecast_steps)

print("Actual_Bond_Yield Forecast:", actual_forecast.values)
print("Predicted_Bond_Yield Forecast:", predicted_forecast.values)