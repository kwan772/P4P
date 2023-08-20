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
# scaler = MinMaxScaler()
scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)
num_feature = 7

# Prepare data in sequences
X, y = [], []
for i in range(1, len(scaled_data)):
    X.append(scaled_data[i-1])
    y.append(scaled_data[i, 3])  # Assuming 'Close' price is the target

X, y = np.array(X), np.array(y)
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

from kerastuner import HyperModel, RandomSearch


# Define a HyperModel for the tuner
class StockPredictionHyperModel(HyperModel):

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_1', 30, 100, 10),
                        activation='relu', input_shape=self.input_shape))
        for i in range(hp.Int('num_layers', 1, 4)):
            model.add(Dense(units=hp.Int(f'units_{i + 2}', 20, 80, 10),
                            activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error')
        return model


hypermodel = StockPredictionHyperModel(input_shape=(num_feature,))

# Use the RandomSearch tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=5,  # Number of model variations to test
    seed=42,
    directory='random_search',
    project_name='stock_prediction'
)

# Search for the best model
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32)

# Get the best model
best_model = tuner.get_best_models(1)[0]

# Evaluate the best model
val_loss = best_model.evaluate(X_val, y_val)
print(f"Validation RMSE of best model: {np.sqrt(val_loss)}")
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)


tuner.results_summary()


