import os

import yfinance as yf
from matplotlib import pyplot as plt
from pytz import timezone
from sqlalchemy import create_engine, text
import pandas as pd

db_connection_str = 'mysql+pymysql://root:'+ os.getenv('DB_PASSWORD') +'@localhost/stock_data'
db_connection = create_engine(db_connection_str)
reddit_df = []

with db_connection.connect() as conn:
    query = f"""
             SELECT created_utc, avg(sentiment_body_score) body_sentiment, avg(sentiment_title_score) title_sentiment, count(id) num_posts, sum(num_comments) num_comments FROM p4p.reddit_posts where symbol = 'GME' group by date(created_utc) order by created_utc;
             """
    reddit_df = pd.read_sql(text(query), conn)


ticker = yf.Ticker("GME")
hist = ticker.history(start="2015-05-28", end="2023-03-24")


# Assuming you have two data frames named df1 and df2

# Reset the index of df2 to convert the DateTimeIndex to a column
hist.reset_index(inplace=True)
# Extract the date portion from 'created_utc' column in reddit_df
reddit_df['date'] = pd.to_datetime(reddit_df['created_utc']).dt.date

# Extract the date portion from 'Date' column in hist
hist['date'] = hist['Date'].dt.date

# print(reddit_df)
# print(hist)


# Perform the merge on 'created_utc' and 'Date'
merged_df = pd.merge(reddit_df, hist, left_on='date', right_on='date', how='right')


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
# Print the merged data frame
print(merged_df.tail(100))




# Set window size to compute moving window synchrony.
r_window_size = 7
# Interpolate missing data.
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = merged_df.select_dtypes(include=numerics)
df_interpolated = newdf.interpolate(method='linear')
# Compute rolling window synchrony
r_price_sentiment = df_interpolated['Close'].rolling(window=r_window_size, center=True).corr(df_interpolated['title_sentiment'])
r_volume_sentiment = df_interpolated['Volume'].rolling(window=r_window_size, center=True).corr(df_interpolated['title_sentiment'])
r_price_num_posts = df_interpolated['Close'].rolling(window=r_window_size, center=True).corr(df_interpolated['num_posts'])
r_volume_num_posts = df_interpolated['Volume'].rolling(window=r_window_size, center=True).corr(df_interpolated['num_posts'])
r_price_num_comments = df_interpolated['Close'].rolling(window=r_window_size, center=True).corr(df_interpolated['num_comments'])
r_volume_num_comments = df_interpolated['Volume'].rolling(window=r_window_size, center=True).corr(df_interpolated['num_comments'])

# Create the plot
plt.figure(figsize=(15,2))
plt.plot(merged_df['date'], r_price_sentiment)
plt.title('Price & Sentiment Correlation')
plt.xlabel('Date')
plt.ylabel('Pearson r')
plt.grid(True)

plt.figure(figsize=(15,2))
plt.plot(merged_df['date'], r_volume_sentiment)
plt.title('Volume & Sentiment Correlation')
plt.xlabel('Date')
plt.ylabel('Pearson r')
plt.grid(True)
plt.show()

plt.figure(figsize=(15,2))
plt.plot(merged_df['date'], r_price_num_posts)
plt.title('Price & Number of Posts Correlation')
plt.xlabel('Date')
plt.ylabel('Pearson r')
plt.grid(True)
plt.show()

plt.figure(figsize=(15,2))
plt.plot(merged_df['date'], r_volume_num_posts)
plt.title('Volume & Number of Posts Correlation')
plt.xlabel('Date')
plt.ylabel('Pearson r')
plt.grid(True)
plt.show()

plt.figure(figsize=(15,2))
plt.plot(merged_df['date'], r_price_num_comments)
plt.title('Price & Number of Comments Correlation')
plt.xlabel('Date')
plt.ylabel('Pearson r')
plt.grid(True)
plt.show()

plt.figure(figsize=(15,2))
plt.plot(merged_df['date'], r_volume_num_comments)
plt.title('Volume & Number of Comments Correlation')
plt.xlabel('Date')
plt.ylabel('Pearson r')
plt.grid(True)
plt.show()