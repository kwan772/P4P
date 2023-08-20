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
sentiment_df.index = sentiment_df.index + pd.DateOffset(days=1)
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
for index,row in result_df.iterrows():
    if row['total_comments'] == 0:
        # print(row)
        zero_count += 1
    total_count+=1

# print(zero_count)
# print(total_count)

# Use High, Low, Open, Close, and Volume for training the model.
data = result_df[['High', 'Low', 'Open', 'Close', 'Volume', 'positive_sentiment_ratio',
           'negative_sentiment_ratio']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
num_feature = 7

from sklearn.ensemble import RandomForestRegressor

# ... [the rest of your imports and data preparation]

# Prepare data for the RandomForest
X_rf, y_rf = [], []
sequence_length = 60
for i in range(sequence_length, len(scaled_data)):
    X_rf.append(scaled_data[i-sequence_length:i].flatten())
    y_rf.append(scaled_data[i, 3])  # Assuming 'Close' price is the target

X_rf, y_rf = np.array(X_rf), np.array(y_rf)
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(X_rf, y_rf, test_size=0.2, shuffle=False)

# from sklearn.model_selection import GridSearchCV
#
# # Define the hyperparameters and their possible values
# param_grid = {
#     'n_estimators': [10, 50, 100, 200],
#     'max_depth': [None, 10, 20, 30, 50],
#     'min_samples_split': [2, 5, 10, 15],
#     'min_samples_leaf': [1, 2, 4, 6, 8],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'bootstrap': [True, False]
# }
#
# # Use the Random Forest Regressor as the model
# rf = RandomForestRegressor()
#
# # Set up GridSearchCV
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
#                            cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
#
# # Fit the model
# grid_search.fit(X_rf, y_rf)
#
# # Get the best parameters
# best_params = grid_search.best_params_
# print(best_params)

param = {'bootstrap': False, 'max_depth': 50, 'max_features': 'sqrt', 'min_samples_leaf': 8, 'min_samples_split': 5, 'n_estimators': 10}
# RandomForest model
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train_rf, y_train_rf)

# Make predictions using RandomForest
predicted_prices_rf = rf_model.predict(X_val_rf)

# Create a dummy array for inverse transformation of RandomForest predictions
dummy_array_predictions_rf = np.zeros((len(predicted_prices_rf), num_feature))
dummy_array_predictions_rf[:, 3] = predicted_prices_rf
inverse_transformed_predictions = scaler.inverse_transform(dummy_array_predictions_rf)[:, 3]

# Create a dummy array for inverse transformation of actual prices
dummy_array_actual = np.zeros((len(y_val_rf), num_feature))
dummy_array_actual[:, 3] = y_val_rf
inverse_transformed_actual = scaler.inverse_transform(dummy_array_actual)[:, 3]

# Calculate the MSE for RandomForest
mse_rf = mean_squared_error(y_val_rf, predicted_prices_rf)
print(f"Random Forest Mean Squared Error: {mse_rf}")

# RMSE for RandomForest
rmse_rf = np.sqrt(mse_rf)
print(f"Random Forest Root Mean Squared Error: {rmse_rf}")

# Print predictions and actual values
for index, actual_val in enumerate(y_val_rf):
    print("Actual: ", scaler.inverse_transform(dummy_array_actual)[index, 3])
    print("Predicted (RandomForest): ", inverse_transformed_predictions[index])
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")

# Calculate the MSE
mse = mean_squared_error(inverse_transformed_actual, inverse_transformed_predictions)
print(f"Mean Squared Error: {mse}")

# RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

