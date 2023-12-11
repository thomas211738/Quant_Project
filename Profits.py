
import yfinance as yf
from Predictions import predict
from DataSet import create_df
import pandas as pd
import csv

def Calc_Profit(ticker, ones_list, investment_amount = 1):

    invested = 0
    profit = 0

    company_info = yf.Ticker(ticker)
    data = company_info.history('5y')
    data = data.iloc[-len(ones_list):]

    j = 0
    for index, row in data.iterrows():

        if ones_list[j][0] == 1:
            invested += investment_amount
            profit += (investment_amount/row['Open']) * (row['Close'] - row['Open'])

        j += 1

    return (invested, profit)


companies = pd.read_csv('Resources/S&P500.csv')
company_names = companies.iloc[:,1].to_list()
company_tickers = companies.iloc[:,2].to_list()


csv_list = []
for i in range(246):
    print(f"Predicting {i}: {company_names[i]}")
    try:
        company_data = create_df(company_tickers[i])
        ones_list, precision = predict(company_data, company_names[i])
        investment_amount, profit = Calc_Profit(company_tickers[i], ones_list)
    except ValueError as e:
        continue

    data = {
        "company_name": company_names[i],
        "company_ticker": company_tickers[i],
        "model_precision": precision,
        "investment_amount": investment_amount,
        "profit": profit
    }
    csv_list.append(data)

csv_file_path = 'Company_Model_Info.csv'
fields = csv_list[0].keys()

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=fields)
    csv_writer.writeheader()
    csv_writer.writerows(csv_list)