
from sklearn.metrics import precision_score
from DataSet import create_df
import math
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import csv

companies = pd.read_csv('Resources/S&P500.csv')
company_names = companies.iloc[:,1].to_list()
company_tickers = companies.iloc[:,2].to_list()

def predict(data,company_name):
    comuns_to_scale = ['ENGULFING', 'RSI', 'MACD_Histogram', 'Bollinger_Band', 'Stochastic_Oscillator',
                       'ichimoku_cloud_signals', 'obv_signal', 'williams_percent_r', 'aroon_indicator']
    scaler = MinMaxScaler(feature_range=(0, 1))
    prediction_days = 100

    test_data = scaler.fit_transform(data[comuns_to_scale])
    test_data = test_data[math.ceil(len(test_data) * .8) - prediction_days:, :]

    target_test = scaler.fit_transform(data['Target'].values.reshape(-1, 1))
    target_test = target_test[math.ceil(len(target_test) * 0.8):, :]
    x_test = []

    for i in range(prediction_days, len(test_data)):
        x_test.append(test_data[i - prediction_days: i, :])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 9))
    y_test = target_test

    highest_p = 0
    model = tf.keras.models.load_model(f'Company Models/{company_name}')
    prediction = model.predict(x_test)

    max_p = 0

    high = max(prediction) - 0.02

    for i in np.arange(0.46, high, 0.01):
        conf = (prediction > i).astype(int)
        score = precision_score(y_test, conf, zero_division=1)

        if score > highest_p:
            max_p = i
            highest_p = score
            ones_list = conf

    ones_list = pd.Series(ones_list.flatten())
    ones_list = (ones_list == 1).sum()

    model_info = {
        'company_name': company_name,
        'confidence': max_p,
        'precision': highest_p,
        'num_ones': ones_list
    }

    return model_info

company_info = []
for i in range(246):
    print(f"Predicting {i}: {company_names[i]}")
    company_data = create_df(company_tickers[i])
    try:
        company_info.append(predict(company_data, company_names[i]))
    except ValueError as e:
        continue

csv_file_path = 'Model_Info.csv'
fields = company_info[0].keys()

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=fields)
    csv_writer.writeheader()
    csv_writer.writerows(company_info)


