import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# Create a connection to your MySQL database
db_connection_str = 'mysql+pymysql://root:' + os.getenv('DB_PASSWORD') + '@localhost/p4p'
engine = create_engine(db_connection_str)

query = f"""
SELECT 
COUNT(*) AS total_comments,
(SUM(CASE WHEN sentiment = 'positive' THEN 1 WHEN sentiment = 'negative' THEN -1 ELSE 0 END) * 1.0) AS net_sentiment,
date(from_unixtime(created_utc)) d FROM p4p.comments_for_certain_symbols where symbol = "gme" group by date(from_unixtime(created_utc));
"""

sentiment_df = None
start_date = "2021-08-01"
end_date = "2022-01-01"

with engine.connect() as conn:
    sentiment_df = pd.read_sql(text(query), conn)

sentiment_df.set_index("d", inplace=True)
sentiment_df.index = pd.to_datetime(sentiment_df.index)
# sentiment_df.index = sentiment_df.index + pd.DateOffset(days=1)
sentiment_df = sentiment_df.sort_index()[start_date:end_date]
# Compute the cumulative sum of net sentiment
sentiment_df['cumulative_sentiment'] = sentiment_df['net_sentiment'].cumsum()

# Compute the 7-day rolling average of net sentiment
sentiment_df['rolling_avg_sentiment'] = sentiment_df['net_sentiment'].rolling(window=7).mean()

# Compute the 7-day EMA for net sentiment
sentiment_df['ema_sentiment'] = sentiment_df['net_sentiment'].ewm(span=7, adjust=False).mean()

symbol = "GME"
df = yf.download(symbol, start=start_date, end=end_date)

# Compute the percentage change for net sentiment and close price
sentiment_df['net_sentiment_pct_change'] = sentiment_df['net_sentiment'].pct_change().fillna(0)
df['Close_pct_change'] = df['Close'].pct_change().fillna(0)

# Plot closed price, rolling average sentiment, and EMA sentiment using matplotlib
fig, ax1 = plt.subplots(figsize=(16,8))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax1.set_xlabel('Date', fontsize=18)
ax1.set_ylabel('Close Price USD ($)', fontsize=18, color="blue")
# ax1.plot(df['Close'], color="blue")
ax1.plot(df['Close_pct_change'], color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

ax2.set_ylabel('Sentiment', fontsize=18)
# ax2.plot(sentiment_df.index, sentiment_df['rolling_avg_sentiment'], color="red", label="7-day Rolling Avg")
# ax2.plot(sentiment_df.index, sentiment_df['ema_sentiment'], color="green", label="7-day EMA")
# ax2.plot(sentiment_df.index, sentiment_df['net_sentiment'], color="green", label="7-day EMA")
ax2.plot(sentiment_df.index, sentiment_df['net_sentiment_pct_change'], color="green", label="7-day EMA")
ax2.tick_params(axis='y')
ax2.legend(loc="upper left")

fig.tight_layout()
plt.title('Close Price with 7-day Rolling Average and 7-day EMA of Sentiment')
plt.show()

