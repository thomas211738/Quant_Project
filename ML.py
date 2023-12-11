
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import pandas as pd
from DataSet import create_df
import math
import numpy as np

companies = pd.read_csv('Resources/S&P500.csv')
company_names = companies.iloc[:,1].to_list()
company_tickers = companies.iloc[:,2].to_list()

def learn(data, company_name, prediction_days = 100):
    scaler = MinMaxScaler(feature_range=(0, 1))
    columns_to_scale = ['ENGULFING', 'RSI', 'MACD_Histogram', 'Bollinger_Band', 'Stochastic_Oscillator',
                        'ichimoku_cloud_signals', 'obv_signal', 'williams_percent_r', 'aroon_indicator']

    target = scaler.fit_transform(data['Target'].values.reshape(-1, 1))
    target = target[:math.ceil(len(target) * .8), :]

    scaled_data = scaler.fit_transform(data[columns_to_scale])
    scaled_data = scaled_data[:math.ceil(len(scaled_data) * .8), :]

    x_train = []
    y_train = target[prediction_days:]

    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, :])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 9))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    model.save(f'Company Models/{company_name}')


for index, (ticker, company) in enumerate(zip(company_tickers,company_names)):
    print(f"Model {index}: {company}")
    company_data = create_df(ticker)
    learn(company_data, company)

