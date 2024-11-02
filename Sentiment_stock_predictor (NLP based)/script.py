import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Function to scrape news headlines related to the stock
def get_news(stock_symbol):
    url = f"https://news.google.com/search?q={stock_symbol}%20stock"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [item.get_text() for item in soup.find_all('a', class_='DY5T1d')[:10]]
    return headlines

# Function to perform sentiment analysis on text data
def analyze_sentiment(text_data):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = [sid.polarity_scores(text)['compound'] for text in text_data]
    return np.mean(sentiment_scores) if sentiment_scores else 0

# Function to fetch stock price data from Yahoo Finance
def get_stock_data(stock_symbol, start, end):
    stock_data = yf.download(stock_symbol, start=start, end=end)
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    return stock_data

# Streamlit app setup
st.title("Sentiment-Based Stock Price Predictor")
stock_symbol = st.text_input("Enter the stock symbol (e.g., AAPL for Apple):", "AAPL")
start_date = st.date_input("Select start date:", pd.to_datetime("2022-01-01"))
end_date = st.date_input("Select end date:", pd.to_datetime("2023-01-01"))

if st.button("Run Analysis"):
    # Step 1: Data Collection
    with st.spinner("Collecting data..."):
        news_headlines = get_news(stock_symbol)
        stock_data = get_stock_data(stock_symbol, start=start_date, end=end_date)
    
    st.subheader("Sample News Headlines")
    st.write("News Headlines:")
    st.write(news_headlines[:5])
    
    # Step 2: Sentiment Analysis
    with st.spinner("Performing sentiment analysis..."):
        news_sentiment = analyze_sentiment(news_headlines)
        st.write(f"Overall Sentiment Score: {news_sentiment}")
    
    # Step 3: Data Preparation for Modeling
    stock_data['Sentiment Score'] = news_sentiment  # Apply the same sentiment score daily for simplicity
    stock_data = stock_data.dropna()
    
    # Feature Engineering
    stock_data['Sentiment Lagged'] = stock_data['Sentiment Score'].shift(1)
    stock_data.dropna(inplace=True)

    X = stock_data[['Sentiment Lagged']]
    y = stock_data['Daily Return']
    
    # Step 4: Machine Learning Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Step 5: Model Evaluation
    mse = mean_squared_error(y_test, predictions)
    st.write(f"Mean Squared Error: {mse:.4f}")
    
    # Step 6: Visualization
    st.subheader("Stock Price vs Sentiment Trend")
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data['Adj Close'], label="Stock Price", color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price", color='blue')
    ax2 = ax.twinx()
    ax2.plot(stock_data.index, stock_data['Sentiment Score'], label="Sentiment Score", color='red')
    ax2.set_ylabel("Sentiment Score", color='red')
    st.pyplot(fig)

    st.subheader("Prediction vs Actual Returns")
    fig, ax = plt.subplots()
    ax.plot(y_test.index, y_test, label="Actual Returns", color='blue')
    ax.plot(y_test.index, predictions, label="Predicted Returns", color='orange')
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    st.pyplot(fig)
