import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sqlalchemy import create_engine, text
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Create a connection to your MySQL database
db_connection_str = 'mysql+pymysql://root:' + os.getenv('DB_PASSWORD') + '@localhost/p4p'
engine = create_engine(db_connection_str)

query = f"""
SELECT 
COUNT(*) AS total_comments,
(SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) * author_weight) AS positive_sentiment_ratio,
(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) * author_weight) AS negative_sentiment_ratio,
date(from_unixtime(created_utc)) d FROM p4p.comments_for_certain_symbols where symbol = "gme" group by date(from_unixtime(created_utc));
"""

start_date = "2020-01-01"
end_date = "2023-01-01"

with engine.connect() as conn:
    sentiment_df = pd.read_sql(text(query), conn)

sentiment_df.set_index("d", inplace=True)
sentiment_df.index = pd.to_datetime(sentiment_df.index)
sentiment_df.index = sentiment_df.index + pd.DateOffset(days=1)
sentiment_df = sentiment_df.sort_index()[start_date:end_date]

tesla = "GME"
df = yf.download(tesla, start=start_date, end=end_date)

result_df = df.join(sentiment_df, how='left').fillna(0.0)

data = result_df[['High', 'Low', 'Open', 'Close', 'Volume', 'positive_sentiment_ratio',
           'negative_sentiment_ratio']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
num_feature = 7

# Prepare data for SVR
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i].flatten())  # Flatten the 2D array into 1D
    y.append(scaled_data[i, 3])

X, y = np.array(X), np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

param={'C': 10, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# param={'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}


# Define the SVR model
model = SVR(**param)
model.fit(X_train, y_train)




# Make predictions
predicted_prices = model.predict(X_val)

# Create a dummy array for inverse transformation of predictions
dummy_array_predictions = np.zeros((len(predicted_prices), num_feature))
dummy_array_predictions[:, 3] = predicted_prices
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

