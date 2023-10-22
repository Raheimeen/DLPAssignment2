import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasRegressor
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score

Datadir = 'D:/FAST_NUCES/7thSemester/DLP/Assignment/Assignment2/Dataset/SamsungDataset/005930.KS.csv'
df = pd.read_csv(Datadir)

df

# Display a summary of the data
summary = df.describe()
print("Data Summary:")
print(summary)

# Convert the 'Date' column to a datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index for the DataFrame
df.set_index('Date', inplace=True)

# Create a plot with Open, Close, High, and Low prices
plt.figure(figsize=(12, 6))

plt.plot(df.index, df['Open'], label='Open', linewidth=2)
plt.plot(df.index, df['Close'], label='Close', linewidth=2)
plt.plot(df.index, df['High'], label='High', linewidth=2)
plt.plot(df.index, df['Low'], label='Low', linewidth=2)

plt.title('Samsung Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)

plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Initialize the Min-Max scaler
scaler = MinMaxScaler()

normalized_col = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df[normalized_col] = scaler.fit_transform(df[normalized_col])

# Define the sequence length
sequence_length = 10  # You can adjust this based on your needs

# Initialize empty lists for input sequences and target values
X = []
y = []

# Iterate through your dataset to create sequences
for i in range(len(df) - sequence_length):
    # Extract the input sequence (historical data)
    input_sequence = df.iloc[i:i + sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values

    # Extract the target value (next day's Close price)
    target_value = df.iloc[i + sequence_length]['Close']

    X.append(input_sequence)
    y.append(target_value)


# Convert the sequences to NumPy arrays
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1)) 
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

plt.plot(y_test, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.title("Actual vs. Predicted Stock Prices")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.show()

def create_lstm_model(num_units):
    model = Sequential()
    model.add(LSTM(num_units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

from sklearn.model_selection import GridSearchCV

# Define the parameters you want to search
param_grid = {
    'num_units': [50, 100],   # Experiment with different numbers of LSTM units
    'epochs': [10, 20],       # Experiment with different numbers of epochs
    'batch_size': [32, 64]    # Experiment with different batch sizes
}

from sklearn.metrics import make_scorer, mean_squared_error
from keras.wrappers.scikit_learn import KerasClassifier

# Create a scoring function for mean squared error
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Create a KerasClassifier based on your model function

est = KerasClassifier(build_fn=create_lstm_model, verbose=0)
# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=est, param_grid=param_grid, scoring=mse_scorer, cv=3,verbose=2, n_jobs=-1)

# Fit the GridSearchCV object on your data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

#Print the best parameters
print(f'Best parameters: {best_params}')

modl = create_lstm_model(100)
modl.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
modl.fit(X_train, y_train, epochs=best_params['epochs'] , batch_size=best_params['batch_size'], validation_data=(X_test, y_test))
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Use the best model for predictions
predictions = modl.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("Best Model - Mean Squared Error:", mse)
print("Best Model - Root Mean Squared Error:", rmse)

plt.plot(y_test, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.title("Actual vs. Predicted Stock Prices")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.show()
