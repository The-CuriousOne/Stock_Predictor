import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objs as go

# Data loading and preprocessing
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

n_years = st.slider('Years of prediction:', 1, 4)
future_period = n_years * 365

data = yf.download(selected_stock, START, TODAY)


st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']].values)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Sequence length for each input sample
X, y = create_sequences(scaled_data, seq_length)

# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=100, batch_size=32, verbose=2)

# Predictions
future_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
future_predictions = []
for i in range(future_period):
    prediction = model.predict(future_seq)
    future_predictions.append(prediction)
    future_seq = np.append(future_seq[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


# Plot predictions
st.subheader('Forecast data')
future_dates = pd.date_range(start=TODAY, periods=future_period)
st.write(pd.DataFrame({'Date': future_dates, 'Forecast': future_predictions.flatten()}))


# Assuming `forecast_dates` and `forecast_data` are defined
forecast_dates = pd.date_range(start=TODAY, periods=future_period)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Data'))
fig1.add_trace(go.Scatter(x=forecast_dates, y=future_predictions.flatten(), name='Forecast', mode='lines'))
fig1.update_layout(title='Forecast Data')
st.plotly_chart(fig1)
