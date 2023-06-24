import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.model_selection import GridSearchCV

# Load data
# Assuming the data is in a file named 'data.csv'
df = pd.read_csv('data.csv')

# Add time difference feature
df['time_diff'] = df['datetime'].diff().dt.total_seconds()
df = df.fillna(0)  # fill the first row of time_diff with 0

# Create moving averages feature
df['moving_avg_3'] = df['can_weight'].rolling(window=3).mean()
df['moving_avg_7'] = df['can_weight'].rolling(window=7).mean()

# Convert date to datetime
df['date'] = pd.to_datetime(df['datetime'])

# Create dummy variables for day of week
df['day_of_week'] = df['date'].dt.dayofweek
df = pd.get_dummies(df, columns=['day_of_week'])

# Drop rows with NaN values (caused by moving average calculation)
df = df.dropna()

# Split data
# 70% for training and 30% for testing
train_size = int(len(df) * 0.7)
train, test = df.iloc[0:train_size], df.iloc[train_size:]

# Normalize features
# Scaling the features to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
test = scaler.transform(test)

# Function to create dataset from time series data
# This function will restructure the data so that the model can be trained to predict the can weight 
# at time t+1 based on the features at time t
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # assuming the target variable (can weight) is the first column
    return np.array(dataX), np.array(dataY)

# Prepare train and test datasets
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# LSTM model
# A LSTM model is a type of recurrent neural network that is well-suited for time series data
def create_lstm(neurons=1):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(look_back, train.shape[1])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# GRU model
# A GRU model is another type of recurrent neural network that is well-suited for time series data
def create_gru(neurons=1):
    model = Sequential()
    model.add(GRU(neurons, input_shape=(look_back, train.shape[1])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Hyperparameters
# The number of neurons in the LSTM and GRU layers will be tuned
neurons = [1, 5, 10]
param_grid = dict(neurons=neurons)

# Cross-validation
# TimeSeriesSplit is a cross-validation technique for time series data
tscv = TimeSeriesSplit(n_splits=10)

# LSTM GridSearchCV
# GridSearchCV is a technique for tuning hyperparameters
lstm = KerasRegressor(build_fn=create_lstm, epochs=100, batch_size=1, verbose=1)
grid_lstm = GridSearchCV(estimator=lstm, param_grid=param_grid, cv=tscv, n_jobs=-1)
grid_result_lstm = grid_lstm.fit(trainX, trainY)

# GRU GridSearchCV
gru = KerasRegressor(build_fn=create_gru, epochs=100, batch_size=1, verbose=1)
grid_gru = GridSearchCV(estimator=gru, param_grid=param_grid, cv=tscv, n_jobs=-1)
grid_result_gru = grid_gru.fit(trainX, trainY)

# Save models
# The trained models will be saved so they can be used later
grid_result_lstm.model.save('lstm_model.h5')
grid_result_gru.model.save('gru_model.h5')

# Make predictions
# The models will predict the can weights for the train and test datasets
trainPredict_lstm = grid_result_lstm.predict(trainX)
testPredict_lstm = grid_result_lstm.predict(testX)
trainPredict_gru = grid_result_gru.predict(trainX)
testPredict_gru = grid_result_gru.predict(testX)

# Convert predictions back to original scale
# The predictions are converted back to the original scale of the can weights
trainPredict_lstm = scaler.inverse_transform(trainPredict_lstm)
trainY = scaler.inverse_transform([trainY])
testPredict_lstm = scaler.inverse_transform(testPredict_lstm)
testY = scaler.inverse_transform([testY])
trainPredict_gru = scaler.inverse_transform(trainPredict_gru)
testPredict_gru = scaler.inverse_transform(testPredict_gru)

# Plot predictions
# The true and predicted can weights are plotted for visual comparison
plt.figure(figsize=(12, 6))
plt.plot(np.concatenate([trainY[0], testY[0]]), label='True')
plt.plot(np.concatenate([trainPredict_lstm, testPredict_lstm]), label='LSTM')
plt.plot(np.concatenate([trainPredict_gru, testPredict_gru]), label='GRU')
plt.xlabel("Samples")
plt.ylabel("Can Weight")
plt.legend()
plt.show()

# Load saved models
# The saved models can be loaded with the load_model function from Keras
lstm_model = load_model('lstm_model.h5')
gru_model = load_model('gru_model.h5')
