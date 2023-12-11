
import pandas as pd
import yfinance as yf

model_data = pd.read_csv('Company_Model_Info.csv')
precision_avg = model_data['model_precision'].mean()
model_roi = ((model_data['investment_amount'] - 1).mean())

data_by_precision = model_data.sort_values(by='investment_amount', ascending=False)

sp_data = yf.Ticker('^GSPC')
sp_data = sp_data.history('4y')
sp_data['Tommorow'] = sp_data['Close'].shift(-1)
sp_data['Target'] = (sp_data['Tommorow'] > sp_data['Close']).astype(int)
sp_data = sp_data.drop(sp_data.index[-1])
sp_avg = sp_data['Target'].sum() / len(sp_data)
sp_roi = ((sp_data.iloc[-1]['Open'] / sp_data.iloc[0]['Open']) - 1)

print(f'Model Average Precision For Last 4 Years: {round(100 * precision_avg,2)}%')
print(f'Model ROI For Last 4 Years: {round(100 * model_roi,2)}%')
print('\n')
print(f'S&P 500 Average Precision For Last 4 Years: {round(100 * sp_avg,2)}%')
print(f'S&P 500 ROI For Last 4 Years: {round(100 * sp_roi,2)}%')
