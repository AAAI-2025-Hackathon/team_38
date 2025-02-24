import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta

fred = Fred(api_key='')

def get_closest_data(series, date):
    """
    Fetch the closest available data point for the given date.
    """
    series = series.dropna().sort_index()
    if date in series.index:
        return series.loc[date]
    else:
        closest_dates = series.index[series.index <= date]
        return series.loc[closest_dates[-1]] if len(closest_dates) > 0 else None

date_range = pd.date_range(start="2014-12-01", end="2024-12-01", freq='ME')  

data = []

for date in date_range:
    print(f"Fetching data for {date.date()}...")  
    
    row = {
        'Date': date
    }
    
    row['US_10Y_Yield'] = get_closest_data(fred.get_series('DGS10'), date)  
    row['AAA_Bond_Yield'] = get_closest_data(fred.get_series('AAA'), date)  
    row['BAA_Bond_Yield'] = get_closest_data(fred.get_series('BAA'), date)  
    row['Junk_Bond_Yield'] = get_closest_data(fred.get_series('BAMLH0A0HYM2'), date)  
    
    data.append(row)

df = pd.DataFrame(data)

df.to_csv("bonds_10yr_weekly_data.csv", index=False)

print("Data collection complete.")
