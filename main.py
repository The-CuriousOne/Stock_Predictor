import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    data_reset_index = data.reset_index()  # Reset index here
    fig.add_trace(go.Scatter(x=data_reset_index['Date'], y=data_reset_index['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data_reset_index['Date'], y=data_reset_index['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'Close']])

# Reshape scaled_data[-1:, :] to match the dimension of the acquired data
scaled_data_last_row = scaled_data[-1:, :].repeat(period + 1, axis=0)  # Repeat the last row to match the number of future dates

# Linear Regression model
model = LinearRegression()

# Train the model
X = np.arange(len(data)).reshape(-1, 1)  # Reshape for single feature
y = scaled_data[:, 1]  # Predicting the Close price
model.fit(X, y)

# Forecasting
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=period + 1)[1:]
X_future = np.arange(len(data), len(data) + len(future_dates)).reshape(-1, 1)
predicted_prices_scaled = model.predict(X_future)

# Inverse transform to get original prices
predicted_prices = scaler.inverse_transform(np.concatenate((scaled_data_last_row, predicted_prices_scaled.reshape(-1, 1)), axis=1))

# Plot forecast data
st.subheader('Forecast Data')
forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=period + 1)[1:]
forecast_data = pd.Series(predicted_prices.flatten(), index=forecast_dates)
st.write(forecast_data)

st.write('Forecast Data')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical Data'))
fig1.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data.values, name='Forecast', mode='lines'))
fig1.update_layout(title='Linear Regression Forecast')
st.plotly_chart(fig1)
