import os

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
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler

# Create a connection to your MySQL database
db_connection_str = 'mysql+pymysql://root:' + os.getenv('DB_PASSWORD') + '@localhost/p4p'
engine = create_engine(db_connection_str)

query = f"""
SELECT 
COUNT(*) AS total_comments,
(SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) * 1.0) AS positive_sentiment_ratio,
(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) * 1.0) AS negative_sentiment_ratio,
date(from_unixtime(created_utc)) d FROM p4p.comments_for_certain_symbols where symbol = "tsla" group by date(from_unixtime(created_utc));
"""

sentiment_df = None
start_date = "2020-01-01"
end_date = "2023-01-01"

with engine.connect() as conn:
    sentiment_df = pd.read_sql(text(query), conn)

sentiment_df.set_index("d", inplace=True)
sentiment_df.index = pd.to_datetime(sentiment_df.index)
# sentiment_df.index = sentiment_df.index + pd.DateOffset(days=1)
sentiment_df = sentiment_df.sort_index()[start_date:end_date]

# print(sentiment_df)

tesla = "TSLA"

# Fetch data
df = yf.download(tesla, start=start_date, end=end_date)
# df = df['2017-12-30':'2023-01-02']

# pd.set_option("display.max_columns", None)

# print(df)
total_count = 0
zero_count = 0

result_df = df.join(sentiment_df, how='left').fillna(0.0)

# print(zero_count)
# print(total_count)

# Use High, Low, Open, Close, and Volume for training the model.
data = result_df[['High', 'Low', 'Open', 'Close', 'Volume', 'positive_sentiment_ratio',
           'negative_sentiment_ratio']].values
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)
num_feature = 7

# Prepare data in sequences
X, y = [], []
for i in range(10, len(scaled_data)):
    X.append(scaled_data[i-10:i].flatten())
    y.append(scaled_data[i, 3])  # Assuming 'Close' price is the target

X, y = np.array(X), np.array(y)
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the LSTM model
model = Sequential()
model.add(Dense(units=50, activation="relu"))
model.add(Dense(units=20, activation="relu"))
model.add(Dense(units=20, activation="relu"))
# model.add(Dense(units=10, activation="relu"))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# hyperparameters = {'units_1': 50, 'num_layers': 2, 'units_2': 80, 'learning_rate': 0.001, 'units_3': 50, 'units_4': 40, 'units_5': 20}
#
# model = Sequential()
#
# # First layer always exists
# model.add(Dense(units=hyperparameters['units_1'], activation="relu"))
#
# # You mentioned 'num_layers': 2, so let's add only 2 more layers.
# # For clarity, I'm considering the first Dense layer you always add as not being counted in the 'num_layers'.
# model.add(Dense(units=hyperparameters['units_2'], activation="relu"))
# model.add(Dense(units=hyperparameters['units_3'], activation="relu"))
#
# # Output layer
# model.add(Dense(units=1))
#
# # Compile the model using the given learning rate
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']), loss='mean_squared_error')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

# Make predictions
predicted_prices = model.predict(X_val)

# Create a dummy array for inverse transformation of predictions
dummy_array_predictions = np.zeros((len(predicted_prices), num_feature))
dummy_array_predictions[:, 3] = predicted_prices[:, 0]
inverse_transformed_predictions = scaler.inverse_transform(dummy_array_predictions)[:, 3]

# Create a dummy array for inverse transformation of actual prices
dummy_array_actual = np.zeros((len(y_val), num_feature))
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
