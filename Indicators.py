
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import math
import csv
import os
import ast

def RSI(data, n=14, column='Close'):
    # Calculate price changes
    delta = data[column].diff()

    # Get gains and losses
    gains = delta.mask(delta < 0, 0)
    losses = -delta.mask(delta > 0, 0)

    # Calculate average gains and losses
    average_gain = gains.rolling(n).mean()
    average_loss = losses.rolling(n).mean()

    # Calculate relative strength (RS)
    rs = average_gain / average_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    rsi = np.where(rsi > 70, 1, np.where(rsi < 30, -1, 0))

    return rsi


def SMA(data, period=30, column='Close'):
    return data[column].rolling(window=period).mean()


def obv_signal(data, period=30):
    avg_volume = data['Volume'].rolling(window=period).mean()
    obv = data['Volume'] > avg_volume
    return obv.astype(int)


def EMA(data, period=14, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()


def LINEARREG(df, period=30, column='Close'):
    closing_prices = df[column].values
    x = np.arange(len(closing_prices))
    # x = np.reshape(x, (-1, 1))

    # Perform linear regression
    model = np.polyfit(x[-period:], closing_prices[-period:], deg=1)

    slope = model[0]
    intercept = model[1]

    line = slope * x + intercept

    return pd.Series(line[-period:], index=df.index[-period:])


def MACD(df, close_column='Close', fast_period=12, slow_period=30, signal_period=9):
    closing_prices = df[close_column].values

    macd_df = pd.DataFrame()

    # Calculate the fast and slow exponential moving averages (EMAs)
    fast_ema = EMA(df, fast_period)
    slow_ema = EMA(df, slow_period)

    # Calculate the MACD line
    macd_df['macd_line'] = fast_ema - slow_ema
    macd_line = fast_ema - slow_ema

    # Calculate the signal line (trigger line) using the signal_period and the MACD line
    signal_line = EMA(macd_df, signal_period, 'macd_line')

    # Calculate the MACD histogram
    histogram = macd_line - signal_line

    # Create DataFrame with MACD values
    macd_df = pd.DataFrame({'MACD': macd_line, 'Signal': signal_line, 'Histogram': histogram}, index=df.index)

    return macd_df


def Bollinger_band(data_frame, window_size=30, num_std=2):
    rolling_mean = data_frame['Close'].rolling(window=window_size).mean()
    rolling_std = data_frame['Close'].rolling(window=window_size).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    signals = np.where(data_frame['Close'] < lower_band, 1, np.where(data_frame['Close'] > upper_band, -1, 0))

    # Create a NaN-filled array with the same length as the data
    nan_array = np.empty(len(data_frame))
    nan_array[:] = np.nan

    # Replace the appropriate positions with the computed signals
    nan_array[-len(signals):] = signals

    return nan_array


def stochastic_oscillator(data_frame, period=15):
    # Calculate the highest closing price and lowest closing price over the given period
    data_frame['Highest Close'] = data_frame['Close'].rolling(window=period).max()
    data_frame['Lowest Close'] = data_frame['Close'].rolling(window=period).min()

    # Calculate the %K and %D values for Stochastic Oscillator
    data_frame['%K'] = (data_frame['Close'] - data_frame['Lowest Close']) / (
                data_frame['Highest Close'] - data_frame['Lowest Close']) * 100
    data_frame['%D'] = data_frame['%K'].rolling(window=3).mean()

    # Initialize a list to store the results (1, -1, or 0)
    results = []

    # Determine the signal (1, -1, or 0) based on %K and %D values
    for i in range(len(data_frame)):
        if data_frame['%K'][i] > data_frame['%D'][i]:
            results.append(1)  # Buy signal
        elif data_frame['%K'][i] < data_frame['%D'][i]:
            results.append(-1)  # Sell signal
        else:
            results.append(0)  # Neutral signal

    # Remove the intermediate columns
    data_frame.drop(columns=['Highest Close', 'Lowest Close', '%K', '%D'], inplace=True)

    return results


def ichimoku_cloud_signals(data):
    # Extract high and low prices from the data
    highs = data['High']
    lows = data['Low']

    # Calculate Tenkan-sen (Conversion Line)
    tenkan_sen = (highs.rolling(window=9).max() + lows.rolling(window=9).min()) / 2

    # Calculate Kijun-sen (Base Line)
    kijun_sen = (highs.rolling(window=26).max() + lows.rolling(window=26).min()) / 2

    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(30)

    # Calculate Senkou Span B (Leading Span B)
    senkou_span_b = (highs.rolling(window=52).max() + lows.rolling(window=52).min() / 2).shift(30)

    # Calculate Kumo (Cloud)
    cloud = senkou_span_a - senkou_span_b

    # Calculate the momentum (difference between current closing price and Chikou Span)
    momentum = data['Close'] - data['Close'].shift(-30)

    # Calculate the Chikou Span (Lagging Span) by shifting the closing prices back by "period" time periods
    chikou_span = data['Close'].shift(-30)

    # Determine the trading signal based on the position of the closing price relative to the Cloud and the Chikou Span
    signals = []

    for i in range(len(data)):
        if data['Close'][i] > cloud[i] and data['Close'][i] > chikou_span[i]:
            signals.append(1)  # Buy signal
        elif data['Close'][i] < cloud[i] and data['Close'][i] < chikou_span[i]:
            signals.append(-1)  # Sell signal
        else:
            signals.append(0)  # No signal

    return signals


def williams_percent_r(df, period=14):
    high = df['High'].rolling(window=period).max()
    low = df['Low'].rolling(window=period).min()
    close = df['Close']

    will_r = (high - close) / (high - low) * -100

    result = []
    for value in will_r:
        if value < -80:
            result.append(1)
        elif value > -20:
            result.append(-1)
        else:
            result.append(0)

    return result


def aroon_indicator(df, period=14):
    up = 100 * (period - df['High'].rolling(period).apply(lambda x: x.argmax())) / period
    down = 100 * (period - df['Low'].rolling(period).apply(lambda x: x.argmin())) / period

    signal = np.where(up > down, 1, np.where(down > up, -1, 0))

    return signal


def ENGULFING(df):
    list = [0]
    for i in range(len(df) - 1):
        prev_candle = df.iloc[i]
        curr_candle = df.iloc[i + 1]

        if prev_candle['Open'] > prev_candle['Close'] and curr_candle['Close'] > curr_candle['Open'] and \
                prev_candle['Close'] < curr_candle['Open'] and prev_candle['Open'] > curr_candle['Close']:
            list.append(-1)
        elif prev_candle['Open'] < prev_candle['Close'] and curr_candle['Close'] < curr_candle['Open'] and \
                prev_candle['Close'] > curr_candle['Open'] and prev_candle['Open'] < curr_candle['Close']:
            list.append(1)
        else:
            list.append(0)
    return list

