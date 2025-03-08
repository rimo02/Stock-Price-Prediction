import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


scaler = MinMaxScaler(feature_range=(0, 1))
seq_length = 60

model = Sequential([
    Input(shape=(seq_length, 1)),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(100, return_sequences=True),
    Dropout(0.2),
    GRU(100, return_sequences=True),
    Dropout(0.2),
    GRU(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

st.title("Stock Price Prediction with LSTM and GRU")

stock = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def train_and_predict(stock):
    data = yf.download(stock, period='5y')['Close']

    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    scaled_prices = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = create_sequences(scaled_prices, seq_length)
    X_train, X_test = X[:train_size-seq_length], X[train_size-seq_length:]
    y_train, y_test = y[:train_size-seq_length], y[train_size-seq_length:]

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    y_pred_lstm = model.predict(X_test)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

    last_sequence = scaled_prices[-seq_length:]
    last_sequence = last_sequence.reshape((1, seq_length, 1))
    next_price = model.predict(last_sequence)
    next_price = scaler.inverse_transform(next_price)


    plt.figure(figsize=(14, 6))
    plt.plot(data.index[train_size:], test,
             label='Actual Prices', color='blue')
    plt.plot(data.index[train_size:], y_pred_lstm,
             label='LSTM Predictions', color='green')
    plt.legend()
    plt.title(f'Stock Price Prediction for {stock}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid()
    st.pyplot(plt)

    st.markdown(
        f"### 📊 Predicted Next Day Price for **{stock}**: **${next_price}**")


if stock:
    
    try:
        train_and_predict(stock)
    except Exception as e:
        st.error(f"An error occurred: {e}")
