#importing lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

#setting date column as index
df=pd.read_csv(r"X:\Project\TSA\TSA\RELIANCE.NS.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.set_index('Date',inplace=True)
del df['Adj Close']

df.shape

#plotting candlestick plot 
import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x=df.index,
                       open=df.Open, high=df.High,
                       low=df.Low, close=df.Close)])

fig.show()

#calculating moving average
Moving_Average_Day = [50, 100, 200]
for Moving_Average in Moving_Average_Day:
  for company in df:
    column_name = f'Moving Average for {Moving_Average} days'
    df[column_name] = df["Close"].rolling(Moving_Average).mean()
    
#plotting moving average
plt.figure(figsize=(20,8))
plt.plot(df.index, df["Close"])
plt.plot(df.index, df["Moving Average for 50 days"],color='red',label='MA for 50 days')
plt.plot(df.index, df["Moving Average for 100 days"],color='green',label='MA for 100 days')
plt.plot(df.index, df["Moving Average for 200 days"],color='orange',label='MA for 200 days')
plt.legend()

#plotting bollinger band
rolling_mean = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()
upper_band = rolling_mean + (rolling_std * 2)
lower_band = rolling_mean - (rolling_std * 2)
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(df.index, df['Close'], label='Close')
ax.plot(rolling_mean.index, rolling_mean, label='Rolling Mean')
ax.fill_between(rolling_mean.index, upper_band, lower_band, alpha=0.4, color='gray', label='Bollinger Bands')
ax.legend()
plt.show()

#scaling data
scaler = StandardScaler()
scaler = scaler.fit(df)
df_s = scaler.transform(df)
len(df_s)
window_size=7

#splitting data into train and test dataset
train_size = int(len(df_s) * 0.8)
train_data = df_s[:train_size]
test_data = df_s[train_size-window_size:]

#function for window size
def df_to_x_y(data, window_size=7):
  
  X = []
  y = []
  for i in range(len(data)-window_size):
    row = [r for r in data[i:i+window_size]]
    X.append(row)
    label = [data[i+window_size][3]]
    y.append(label)
  return np.array(X), np.array(y)

x_train,y_train = df_to_x_y(train_data)
x_test,y_test=df_to_x_y(test_data)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#CNN-LSTM model
model = Sequential()


model.add(Conv1D(filters=256, kernel_size= 1, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
#model.add(Conv1D(filters=128, kernel_size= 1,  activation='relu'))
model.add(MaxPooling1D(pool_size=5, padding='valid'))
model.add(Conv1D(filters=64,  kernel_size= 1, activation='relu'))

model.add(LSTM(units=100, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(units=75, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))

model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.summary()

history=model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat=model.predict(x_test)

yp=yhat.reshape(yhat.shape[0],1)
len(yp)

test_RMSE= np.sqrt(mean_squared_error(y_test, yp))
test_MAE= mean_squared_error(y_test, yp )

print(f"Test RMSE: {test_RMSE}")
print(f"Test MAE: {test_MAE}")

# create empty table with 12 fields
trainPredict_dataset_like = np.zeros(shape=(len(yp), 5) )
# put the predicted values in the right field
trainPredict_dataset_like[:,0] = yp[:,0]
# inverse transform and then select the right field
trainPredict = scaler.inverse_transform(trainPredict_dataset_like)[:,0]

plt.figure(figsize=(20,6))
plt.plot(df.index[train_size:],df.iloc[train_size:,0], label='Actual')
plt.plot(df.index[train_size:],trainPredict, label='Actual')

plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error