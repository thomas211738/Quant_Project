
import pandas as pd
import yfinance as yf

model_data = pd.read_csv('Model_Info.csv')
total_num_ones = model_data['num_ones'].sum()
precision_avg = (model_data['precision'] * (model_data['num_ones']/total_num_ones)).sum()
data_by_precision = model_data.sort_values(by='precision', ascending=False)

print(f'Model Average Precision: {round(100 * precision_avg,2)}%')

sp_data = yf.Ticker('^GSPC')
sp_data = sp_data.history('2y')
sp_data['Tommorow'] = sp_data['Close'].shift(-1)
sp_data['Target'] = (sp_data['Tommorow'] > sp_data['Close']).astype(int)
sp_data = sp_data.drop(sp_data.index[-1])

sp_avg = sp_data['Target'].sum() / len(sp_data)
sp_roi = ((sp_data.iloc[-1]['Open'] / sp_data.iloc[0]['Open']) - 1)

print(f'S&P 500 Average Precision For Last 2 Years: {round(100 * sp_avg,2)}%')
print(f'S&P 500 ROI since Dec 2021: {round(100 * sp_roi,2)}%')