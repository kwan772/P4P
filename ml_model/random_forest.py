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
from sklearn.ensemble import RandomForestRegressor

# Fetch data for Tesla over a specific period (e.g., the past 5 years)
tesla = "TSLA"
start_date = "2020-01-01"
end_date = "2023-01-01"
df = yf.download(tesla, start=start_date, end=end_date)

pd.set_option("display.max_columns", None)
# Use High, Low, Open, Close, and Volume for training the model.
data = df[['High', 'Low', 'Open', 'Close', 'Volume']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare data without sequences (Random Forest doesn't need sequential input like LSTM)
X_rf, y_rf = [], []
for i in range(60, len(scaled_data)):
    X_rf.append(scaled_data[i-60:i].flatten())  # Flattening the sequences
    y_rf.append(scaled_data[i, 3])

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
#
param={
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True
}
rf = RandomForestRegressor(**param)
rf.fit(X_train_rf, y_train_rf)
predictions_rf = rf.predict(X_val_rf)

# Make predictions
predicted_prices = rf.predict(X_val_rf)

# Create a dummy array for inverse transformation of predictions
dummy_array_predictions = np.zeros((len(predicted_prices), 5))
dummy_array_predictions[:, 3] = predicted_prices
# dummy_array_predictions[:, 3] = predicted_prices[:, 0]
inverse_transformed_predictions = scaler.inverse_transform(dummy_array_predictions)[:, 3]

# Create a dummy array for inverse transformation of actual prices
dummy_array_actual = np.zeros((len(y_val_rf), 5))
dummy_array_actual[:, 3] = y_val_rf
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

