import yfinance as yf
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


df = pd.read_csv('/Users/siheonjung/Desktop/psu/spring 2024/ds440/7/code(lstm)/TSLA.csv')
################ need to change file path ################

def load_data(company):
    dataframe = df.copy()
    dataframe.insert(0, 'Name', company)
    del dataframe['Adj Close']
    dataframe = dataframe[['Name', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
    return dataframe

COMPANY = 'TSLA'
################ need to change COMPANY ################

data = load_data(company = COMPANY)

print(data)