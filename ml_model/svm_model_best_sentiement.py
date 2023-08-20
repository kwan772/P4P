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

from sklearn.metrics import make_scorer

def rmse_scorer(true, predicted):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return -rmse

negative_rmse = make_scorer(rmse_scorer)


# Create a connection to your MySQL database
db_connection_str = 'mysql+pymysql://root:' + os.getenv('DB_PASSWORD') + '@localhost/p4p'
engine = create_engine(db_connection_str)

query = f"""
SELECT 
COUNT(*) AS total_comments,
(SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) * 1.0) AS positive_sentiment_ratio,
(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) * 1.0) AS negative_sentiment_ratio,
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


# 0.23554120822656083 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23554120822656083 {'C': 10, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23554120822656083 {'C': 10, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23554120822656083 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.2355412082265608 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.2355412082265608 {'C': 10, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.2355412082265608 {'C': 10, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.2355412082265608 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.2355412082265608 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.2355412082265608 {'C': 10, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.23410045102137297 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 1e-05}
# 0.23410045102137297 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 1e-05}
# 0.2340902155081092 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}
# 0.2340902155081092 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.0001}
# 0.2340544566503859 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.001}
# 0.23405445665038588 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2338646418778706 {'C': 1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2338646418778706 {'C': 1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.2333791604290331 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 1e-05}
# 0.2333791604290331 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.0001}
# 0.2331878442979877 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.001}
# 0.23298939119827433 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}
# 0.23297725868738645 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 1e-05}
# 0.23293659689329113 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 1e-05}
# 0.23293659689329113 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 1e-05}
# 0.23293536184147037 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 1e-05}
# 0.23293536184147037 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 1e-05}
# 0.23292944508814015 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.0001}
# 0.23292944508814015 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}
# 0.2329261814300576 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.0001}
# 0.2329261814300576 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}
# 0.23290855378099826 {'C': 100, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23290855378099826 {'C': 100, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23290855378099826 {'C': 100, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23290855378099826 {'C': 100, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23290855378099826 {'C': 100, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23290855378099826 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.23290206037641567 {'C': 100, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.23290206037641567 {'C': 100, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.23290206037641567 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.23290206037641567 {'C': 100, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.23290206037641567 {'C': 100, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.23290206037641567 {'C': 100, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.23286002714974652 {'C': 100, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.23286002714974652 {'C': 100, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.23286002714974652 {'C': 100, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.23286002714974652 {'C': 100, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.23286002714974652 {'C': 100, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.23286002714974652 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.23284087948469764 {'C': 100, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.23284087948469764 {'C': 100, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.23284087948469764 {'C': 100, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.23284087948469764 {'C': 100, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.23284087948469764 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.23284087948469764 {'C': 100, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.23282119871150128 {'C': 100, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.23282119871150128 {'C': 100, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.23282119871150128 {'C': 100, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.23282119871150128 {'C': 100, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.23282119871150128 {'C': 100, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.23282119871150128 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.23282119871150128 {'C': 100, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.23282119871150128 {'C': 100, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.23282119871150128 {'C': 100, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.23282119871150128 {'C': 100, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.23282119871150128 {'C': 100, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.23282119871150128 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.23282046630026124 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.001}
# 0.23282046630026124 {'C': 1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.001}
# 0.23279057404410963 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.001}
# 0.23279057404410963 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.001}
# 0.2327586600503209 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': True, 'tol': 1e-05}
# 0.2327586600503209 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': False, 'tol': 1e-05}
# 0.23274585826173597 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}
# 0.23274585826173597 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': False, 'tol': 0.0001}
# 0.23273995337165124 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': False, 'tol': 0.001}
# 0.23273995337165124 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': True, 'tol': 0.001}
# 0.23271693867432264 {'C': 100, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.001}
# 0.22932391458152301 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.22932391458152301 {'C': 10, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.22932391458152301 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.22932391458152301 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.22932391458152301 {'C': 10, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.22932391458152301 {'C': 10, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 1e-05}
# 0.2293226532064946 {'C': 10, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.2293226532064946 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.2293226532064946 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.2293226532064946 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.2293226532064946 {'C': 10, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.2293226532064946 {'C': 10, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 1e-05}
# 0.22930647610512003 {'C': 10, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.22930647610512003 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.22930647610512003 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.22930647610512003 {'C': 10, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.22930647610512003 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.22930647610512003 {'C': 10, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
# 0.22930154264774938 {'C': 10, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.22930154264774938 {'C': 10, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.22930154264774938 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.22930154264774938 {'C': 10, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.22930154264774938 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.22930154264774938 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.0001}
# 0.22916388502163104 {'C': 10, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.22916388502163104 {'C': 10, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.22916388502163104 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.22916388502163104 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.22916388502163104 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.22916388502163104 {'C': 10, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': False, 'tol': 0.001}
# 0.22916388502163101 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.22916388502163101 {'C': 10, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.22916388502163101 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.22916388502163101 {'C': 10, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.22916388502163101 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.22916388502163101 {'C': 10, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.001}
# 0.2273225891637829 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.001}
# 0.2273225891637829 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.001}
# 0.227172193630886 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 1e-05}
# 0.227172193630886 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 1e-05}
# 0.2271710637061165 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.0001}
# 0.2271710637061165 {'C': 10, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}
# 0.22655325229957723 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.0001}
# 0.22655273000054232 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 1e-05}
# 0.22655231272579654 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 1e-05}
# 0.2265514631619612 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}
# 0.22644608540053648 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.001}
# 0.22643159105710983 {'C': 10, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.001}
# 0.22572341071442253 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 1e-05}
# 0.22572341071442253 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 1e-05}
# 0.2257228847143682 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}
# 0.2257228847143682 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.0001}
# 0.2255182287946236 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.001}
# 0.2255182287946236 {'C': 10, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': False, 'tol': 0.001}
# 0.22195377337095162 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22195377337095162 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22192498064182886 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22192498064182886 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22186811943073478 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22186811943073478 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.2218426044958704 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2218426044958704 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22182828952629063 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22181956210189635 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.2209992129356436 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22098054498934186 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.001}
# 0.22093143676932372 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093143676932372 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 1e-05}
# 0.22093099332348834 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.22093099332348834 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': False, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.2207270524296526 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 0.0001}
# 0.22057217837224463 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 1.0, 'degree': 3, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 0.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 0.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 1.0, 'degree': 2, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# 0.22057217837224463 {'C': 0.1, 'coef0': 1.0, 'degree': 4, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True, 'tol': 1e-05}
# nan {'tol': 1e-05, 'shrinking': True, 'kernel': 'poly', 'gamma': 0.01, 'epsilon': 0.001, 'degree': 2, 'coef0': 1.0, 'C': 100}
# nan {'tol': 0.001, 'shrinking': True, 'kernel': 'rbf', 'gamma': 'auto', 'epsilon': 0.001, 'degree': 4, 'coef0': 0.0, 'C': 10}
# nan {'tol': 0.0001, 'shrinking': True, 'kernel': 'linear', 'gamma': 0.1, 'epsilon': 0.001, 'degree': 5, 'coef0': 0.5, 'C': 1}
# nan {'tol': 1e-05, 'shrinking': False, 'kernel': 'sigmoid', 'gamma': 'auto', 'epsilon': 0.001, 'degree': 3, 'coef0': 0.0, 'C': 1}
# nan {'tol': 1e-05, 'shrinking': False, 'kernel': 'poly', 'gamma': 'scale', 'epsilon': 0.01, 'degree': 2, 'coef0': 1.0, 'C': 0.1}
# Your provided parameters
params = {'tol': 1e-05, 'shrinking': False, 'kernel': 'sigmoid', 'gamma': 'auto', 'epsilon': 0.001, 'degree': 3, 'coef0': 0.0, 'C': 1}

# Create the SVR model with the specified parameters
model = SVR(**params)
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



