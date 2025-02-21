import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import os 
from dotenv import load_dotenv

load_dotenv('.env')
api_key=os.getnv("fetch_dataset")
fred = Fred(api_key=api_key)

input_date = '2024-02-18'
input_date = pd.to_datetime(input_date)

def get_closest_data(series, date):
    """
    Fetch the closest available data point for the given date.
    """
    series = series.sort_index()
    if date in series.index:
        return series.loc[date]
    else:
        closest_date = series.index[series.index <= date][-1]
        return series.loc[closest_date]

data = {}

data['US_10Y_Yield'] = get_closest_data(fred.get_series('DGS10'), input_date)

data['AAA_Bond_Yield'] = get_closest_data(fred.get_series('AAA'), input_date)
data['BAA_Bond_Yield'] = get_closest_data(fred.get_series('BAA'), input_date)

data['Junk_Bond_Yield'] = get_closest_data(fred.get_series('J'), input_date)

df = pd.DataFrame(data, index=[input_date])

df.to_csv("bonds.csv")