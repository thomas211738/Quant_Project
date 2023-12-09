from indicators import RSI, SMA, obv_signal, stochastic_oscillator, ichimoku_cloud_signals, aroon_indicator,\
    EMA, ENGULFING, MACD, Bollinger_band, LINEARREG, williams_percent_r
    
import pandas as pd
import yfinance as yf
import numpy as np

def create_df(ticker: str, start_date: str, end_date:str) -> pd.DataFrame:

    data = yf.download(ticker, start=start_date, end=end_date)
    
    data['SMA20'] = data['Close'] / SMA(data, 20)
    data['SMA50'] = data['Close'] / SMA(data, 50)
    data['SMA200'] = data['Close'] / SMA(data, 200)
    data['RSI'] = RSI(data)
    macd = MACD(data)
    data['MACD_Histogram'] = np.where(macd['Histogram'] > 0, 1, -1)
    data['Bollinger_Band'] = Bollinger_band(data)
    data['ENGULFING'] = ENGULFING(data)
    data['Stochastic_Oscillator'] = stochastic_oscillator(data)
    data['ichimoku_cloud_signals'] = ichimoku_cloud_signals(data)
    data['obv_signal'] = obv_signal(data)
    data['williams_percent_r'] = williams_percent_r(data)
    data['aroon_indicator'] = aroon_indicator(data)

    data['Tommorow'] = data['Close'].shift(-1)
    data['Target'] = (data['Tommorow'] > data['Close']).astype(int)

    data = data.copy()
    data = data.dropna(subset=data.columns[data.columns != "Tommorow"])

    return data

data = create_df("AAPL", "2022-01-01","2023-01-01")
print(data.shape)