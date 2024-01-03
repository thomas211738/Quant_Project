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

## Model Analysis

### Accuracy

To test our model's accuracy, we used the last 20% (4y) of our data. We needed to find out when our model predicted the stock going up, and then compare those instances with whether the stock actually went up or not. To do this, we used the precision_score function from sklearn, and found that we had an average precision score of 56.77% for the 250 models. This means that for every time the model predicted the stock to go up, it would go up 56.77% of the time.

### Profits

To determine the profits or ROI someone would gain from applying this model to actual trading, we started by investing $1 in each company. Then, each time the model predicted the price going up, we would invest this $1 for the day. We would then keep updating this investment through the 4 years' worth of data to determine what our final investment is worth and ROI.


We found that the average ROI was 169% over the past 4 years. Our Highest ROI was for Apple, which had an ROI of 450% over the past 4 years. Our lowest ROI was for Coca-Cola, which had an ROI of 96%. Within that same time frame, 

## Conclusion


