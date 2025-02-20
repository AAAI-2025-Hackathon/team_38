import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from fredapi import Fred
from datetime import datetime

# Initialize FRED API (Replace with your FRED API key)
fred = Fred(api_key='242b0976c07592e0fd1536f9948517c9')

# Specify the date for which you want the data
input_date = '2024-02-18'  # Replace with your desired date
input_date = pd.to_datetime(input_date)

# Function to fetch data for the closest available date
def get_closest_data(series, date):
    """
    Fetch the closest available data point for the given date.
    """
    series = series.sort_index()
    if date in series.index:
        return series.loc[date]
    else:
        # Get the closest date before the input date
        closest_date = series.index[series.index <= date][-1]
        return series.loc[closest_date]

# Initialize an empty dictionary to store the data
data = {}

# 1. US Treasury Yields
data['US_10Y_Yield'] = get_closest_data(fred.get_series('DGS10'), input_date)  # 10-Year Treasury Yield

# 2. Corporate Bond Yields by Rating
data['AAA_Bond_Yield'] = get_closest_data(fred.get_series('AAA'), input_date)  # AAA-rated corporate bonds
data['BAA_Bond_Yield'] = get_closest_data(fred.get_series('BAA'), input_date)  # BAA-rated corporate bonds

# 3. High-Yield (Junk) Bond Yield (BB & lower)
data['Junk_Bond_Yield'] = get_closest_data(fred.get_series('J'), input_date)  # High-Yield Index

# Create a DataFrame
df = pd.DataFrame(data, index=[input_date])

# Display the DataFrame
df.to_csv("bonds.csv")