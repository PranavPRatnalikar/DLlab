import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# a. Load dataset
df = pd.read_csv('weather.csv')  # Replace with your file
df = df[['Date', 'Temperature']]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# b. Preprocess: Normalize values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['Temperature']])

# c. Create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 30  # use last 30 days to predict next day
X, y = create_sequences(data_scaled, window_size)

# Split train-test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape input [samples, time_steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# d. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16)

# e. Predict
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# f. Evaluate RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print("RMSE:", rmse)

# g. Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(df.index[-len(y_test):], y_test_inv, label='Actual Temperature')
plt.plot(df.index[-len(y_test):], y_pred_inv, label='Predicted Temperature')
plt.title('LSTM Weather Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
