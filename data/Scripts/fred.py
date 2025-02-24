import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta

fred = Fred(api_key='')

end_date = datetime(2024, 12, 1)
start_date = datetime(2014, 12, 1)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

print(f"Collecting data from {start_date.date()} to {end_date.date()}")

date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
economic_data = pd.DataFrame(index=date_range)

print("Collecting FRED data...")
fred_series = {
    'Inflation_Rate': 'CPIAUCNS',
    'GDP_Growth': 'GDPC1',
    'Unemployment_Rate': 'UNRATE',
    'Fed_Funds_Rate': 'FEDFUNDS',
    'Money_Supply': 'M2SL',
    'Consumer_Confidence': 'UMCSENT',
    'US_10Y_Yield': 'DGS10'
}

for name, code in fred_series.items():
    print(f"  Fetching {name}...")
    series = fred.get_series(code, start_date, end_date)
    
    if name == 'Inflation_Rate':
        series = series.pct_change(periods=12) * 100
    elif name == 'GDP_Growth':
        series = series.pct_change(periods=1) * 100 * 4
    
    if series.index.freq != 'M':
        series = series.resample('M').last()
    
    economic_data[name] = series

print("Collecting market data...")
tickers = {
    'S&P_500': '^GSPC',
    'Crude_Oil': 'CL=F',
    'Gold': 'GC=F', 
    'US_Dollar_Index': 'DX-Y.NYB',
    'INR_USD': 'INR=X',
    'VIX': '^VIX'
}

yf_data = yf.download(list(tickers.values()), start=start_date, end=end_date)

for name, ticker in tickers.items():
    print(f"  Processing {name}...")
    if isinstance(yf_data.columns, pd.MultiIndex):
        if ticker in yf_data['Close'].columns:
            series = yf_data['Close'][ticker].resample('M').last()
            economic_data[name] = series
        else:
            print(f"    Warning: No data found for {ticker}")
    else:
        series = yf_data['Close'].resample('M').last()
        economic_data[name] = series

economic_data = economic_data.fillna(method='ffill', limit=3)

print(f"\nCollected data shape: {economic_data.shape}")
print("\nData preview:")
print(economic_data.head())

csv_filename = f"economic_data_10yr_{end_date.strftime('%Y-%m-%d')}.csv"
economic_data.to_csv(csv_filename)
print(f"\nData saved to {csv_filename}")