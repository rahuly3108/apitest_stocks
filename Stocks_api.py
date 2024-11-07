#!/usr/bin/env python
# coding: utf-8

# In[10]:


from flask import Flask, jsonify
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import skew
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

def process_data():
    # Load the list of 200 stock symbols
    data = pd.read_csv('ind_nifty200list.csv')
    symbols_list = data['Symbol']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    results_list = []

    # Loop through each stock symbol and perform the calculations
    for symbol in symbols_list:
        stock_df = yf.download(symbol + '.NS', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)['Adj Close'].reset_index()
        stock_df.columns = ['Date', symbol]
        stock_df['Return'] = stock_df[symbol].pct_change() * 100
        stock_df = stock_df.dropna()
        if stock_df.empty:
            print(f"After cleaning, no data found for {symbol}. Skipping.")
            continue
        pos_ret = stock_df['Return'][stock_df['Return'] >= 0]
        neg_ret = stock_df['Return'][stock_df['Return'] < 0]
        new_row = {
            'Stock': symbol,
            '% Positive Returns': (len(pos_ret) / len(stock_df)) * 100,
            '% Negative Returns': (len(neg_ret) / len(stock_df)) * 100,
            'Avg Positive Return': pos_ret.mean(),
            'Median Positive Return': pos_ret.median(),
            'Avg Negative Return': neg_ret.mean(),
            'Median Negative Return': neg_ret.median(),
            'SD Positive Return': pos_ret.std(),
            'Skewness Positive Return': skew(pos_ret),
            'SD Negative Return': neg_ret.std(),
            'Skewness Negative Return': skew(neg_ret),
            'Min Positive Return': pos_ret.min(),
            'Max Positive Return': pos_ret.max(),
            'Min Negative Return': neg_ret.min(),
            'Max Negative Return': neg_ret.max()
        }
        results_list.append(new_row)
    
    results_df2 = pd.DataFrame(results_list).round(3)

    # Calculate requested metrics for the new table
    new_df = pd.DataFrame()
    new_df['Stock'] = results_df2['Stock']
    new_df['Percentage Difference'] = (results_df2['% Positive Returns']/abs(results_df2['% Negative Returns']))
    new_df['Average Difference'] = results_df2['Avg Positive Return']/abs(results_df2['Avg Negative Return'])
    new_df['Median Difference'] = results_df2['Median Positive Return']/abs(results_df2['Median Negative Return'])
    new_df['Std Dev Difference'] = results_df2['SD Positive Return']/abs(results_df2['SD Negative Return'])
    new_df['Skewness Difference'] = results_df2['Skewness Positive Return']/abs(results_df2['Skewness Negative Return'])

    # Calculate max-min difference without absolute values for negative returns
    new_df['Max-Min Positive/Negative'] = (results_df2['Max Positive Return'] - results_df2['Min Positive Return']) / \
                                           (results_df2['Max Negative Return'] - results_df2['Min Negative Return'])

    # Optional: Round the results to 3 decimal places
    new_df_rounded = new_df.round(3)

    # Standardize the data for clustering
    Cluster_data = new_df_rounded.drop(columns=['Stock'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(Cluster_data)

    # Choose the number of clusters (K)
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    cluster_labels = kmeans.labels_
    new_df_rounded['Cluster'] = cluster_labels + 1

    return new_df_rounded


@app.route("/")
def get_data():
    # Process the data and return the result as a JSON response
    new_df_rounded = process_data()

    # Convert the result_df to a dictionary and send it as JSON
    return jsonify(new_df_rounded.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:




