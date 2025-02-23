import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('rl_120_months.csv')

keys = [['US_10Y_Yield_Actual', 'US_10Y_Yield'], ['AAA_Bond_Yield_Actual', 'AAA_Bond_Yield'], [
    'BAA_Bond_Yield_Actual', 'BAA_Bond_Yield'], ['Junk_Bond_Yield_Actual', 'Junk_Bond_Yield']]
look_back = 120-36
forecast_steps = 36

df_new = pd.DataFrame()
for k in keys:
    actual_series = df[k[0]]
    predicted_series = df[k[1]]

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

    df_new[k[0]] = actual_forecast.values
    df_new[k[1]] = predicted_forecast.values

df_new.to_csv("Forecasting/arima_rl.csv")
