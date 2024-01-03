# Machine Learning on Technical Analysis Indicators
This project was created by @thomas211738 and @joshleeds

## Overview
We wanted to see if technical analysis indicators were good predictors of a stock's future directionality. Since there are many technical analysis indicators, we decided to use as many as possible and apply them as features to a machine learning model to identify the best combination of technical analysis indicators that result in the best results. We did this for about 250 stocks to test how well the models worked for different stocks and avoid overfitting our model. 

## Technical Analysis Indicators
For this project, we used the following technical analysis indicators as features:

1. Simple Moving Average (SMA)
2. Exponential Moving Average (EMA)
3. Relative Strength Index (RSI)
4. Moving Average Convergence Divergence (MACD)
5. Bollinger Bands
6. Stochastic Oscillator
7. Ichimoku Cloud Signals
8. Aroon Indicator
9. Williams %R
10. On-Balance-Volume (OBV)

## Data Set
This is what one year's worth of data for our data set would look like.
![Visual](TAIEX.png)
As we can see, for each of the technical analysis indicators, they give a value of either a -1, meaning that particular indicator believes the stock will go down, 0 meaning it believes it will stay relatively the same, or a 1, meaning the stock will go up. We also have a Target column on the right, which indicates the stock directionality for the next day, with 0 meaning the stock price went down, and 1 meaning the stock price went up. 

## Machine Learning Model
For the machine learning model, we used an LSTM neural network using keras in tensor flow. We trained 250 different models for the 250 companies that we chose. We first acquired 20 years' worth of stock data for each company and split it with 80% (16y) of the data to be trained on, and 20% (4y) of the data to be tested on. We also used the past 100 days of our data to predict the stock directionality one day in the future. We did this because we thought it would give us more accurate and better results since the model would have more data to analyze and work with. for our x data, we gave the model the past 100 days of our technical analysis indicator's values, and for the y data, we gave the model the stock directionality value for the next day. We also had an EPOCH of 10, meaning the model passed over the 16 years' worth of stock data 10 times. 

## Model Anlysis


