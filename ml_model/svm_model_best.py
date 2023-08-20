import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Fetch data for GameStop over a specific period
tesla = "GME"
start_date = "2020-01-01"
end_date = "2023-01-01"
df = yf.download(tesla, start=start_date, end=end_date)

pd.set_option("display.max_columns", None)
data = df[['High', 'Low', 'Open', 'Close', 'Volume']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare data for SVM
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i].flatten())  # Flatten the 2D array into 1D
    y.append(scaled_data[i, 3])  # Assuming 'Close' price is the target

X, y = np.array(X), np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

params = {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}

# Define the SVR model
model = SVR(**params)  # You can experiment with different kernels like 'rbf'
model.fit(X_train, y_train)

# Make predictions
predicted_prices = model.predict(X_val)

# Create a dummy array for inverse transformation of predictions
dummy_array_predictions = np.zeros((len(predicted_prices), 5))
dummy_array_predictions[:, 3] = predicted_prices
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
