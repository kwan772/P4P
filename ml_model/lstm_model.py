import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import yfinance as yf
from sklearn.metrics import mean_squared_error
import datetime

# Fetch data for Tesla over a specific period (e.g., the past 5 years)
tesla = "GME"
start_date = "2020-01-01"
end_date = "2023-01-01"
df = yf.download(tesla, start=start_date, end=end_date)

pd.set_option("display.max_columns", None)
# Use High, Low, Open, Close, and Volume for training the model.
data = df[['High', 'Low', 'Open', 'Close', 'Volume']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare data in sequences
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i, 3])  # Assuming 'Close' price is the target

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 5))  # Updated to consider 5 features
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 5)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

# Make predictions
predicted_prices = model.predict(X_val)

# Create a dummy array for inverse transformation of predictions
dummy_array_predictions = np.zeros((len(predicted_prices), 5))
dummy_array_predictions[:, 3] = predicted_prices[:, 0]
inverse_transformed_predictions = scaler.inverse_transform(dummy_array_predictions)[:, 3]

# Create a dummy array for inverse transformation of actual prices
dummy_array_actual = np.zeros((len(y_val), 5))
dummy_array_actual[:, 3] = y_val
inverse_transformed_actual = scaler.inverse_transform(dummy_array_actual)[:, 3]


for index, row in enumerate(inverse_transformed_actual):
    print("actual " + str(inverse_transformed_actual[index]))
    print("predicted " + str(inverse_transformed_predictions[index]))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")

# Calculate the MSE
mse = mean_squared_error(inverse_transformed_actual, inverse_transformed_predictions)
print(f"Mean Squared Error: {mse}")

# RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Save the model
model.save("tesla_lstm_model.keras")

