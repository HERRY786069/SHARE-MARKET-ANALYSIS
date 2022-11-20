# SHARE-MARKET-ANALYSIS
#herryjerry047@gmail.com
import numpy as np
import pandas as pd
import matplotlip.pyplot as plt
import pandas_datareader as data 



start = '2010-01-01'
end = '2019-12-31'

df = data.DataReader('AAPL','yahoo',start,end)
df.head()
df.tail()
df = df.reset_index()
df.head()
df = df.drop(['date','adj close'], axis = 1)
df.head()
plt.plot(df.close)
df
ma100 = df.close.rolling(100).mean()
ma100
plt.figure(figsize = (12,6))
plt.plot(df.close)
plt.plot(ma100, 'r')
#this a stregy which invester anaylse the stock ma (moving average)

ma200 = df.close.rolling(200).mean()
ma200

plt.figure(figsize = (12,6))
plt.plot(df.close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

#its to check how many column is used
df.shape
#we are doing 70%data is training and 30%data is on testing
#splitting data into trainig and testing
data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

data_trainig.head()
data_testing.head()

from sklearn.preprocessing import MinMaxscaler
scaler = MinMaxscaler(feature_range=(0,1))

data_training_array = scaler.fit_tramsformer(data_training)
data_training_array

x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
x_train.append(data_training_array.shape[i-100: i])
y_train.append(data_training_array.shape[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
#ML Model
from keras.layers import Dense,dropout , LSTM
from Keras.models import sequential

model = sequential()
model.add(LSTM(units = 50, activation = 'relu', return-sequences = True,
input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return-sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return-sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.compile(optimizer= 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

#runnig data first then write next programkming

model.save('Keras_model.h5')
data_testing.head()
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
final_df.head()
input_data = scaler.fit_transform(final_df)
input_data


input_data.shape

x_train = []
y_train = []

for i in range(100, input_data.shape[0]):
x_test.append(input_data.shape[i-100: i])
y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)

#making Predictions 

y_predicted = model.predict(x_test)

y_predicted.shape

y_test

y_predicted

scaler.scale_
#run upper line code you get a some number copy that in the second line code

scale_factor = 1/0.02099517
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()



#what are we do 
we are predicting price of share by data
by analyzing 10days data we can get 11th days price we predict
this the logic
10days is a xtrain(data)
11th  is a ytrain (prediction)
if you want to predict the 12th date data price
its easy remove 1days from 10days data then it will be 9days data 
then 11th date data in that 9days data you get the 10days of data now 
you can predict the 12th date data
for 13th same goes on
