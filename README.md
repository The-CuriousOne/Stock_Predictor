# Stock Forecast App

This Streamlit application utilizes machine learning techniques to forecast stock prices for selected companies. It leverages historical stock price data obtained from Yahoo Finance using the `yfinance` library. The forecasting model is built using TensorFlow and employs Convolutional Neural Networks (CNNs).

## Functionality

- **Data Loading and Preprocessing**: Historical stock price data from specified companies (Google, Apple, Microsoft, GameStop) is loaded and preprocessed for analysis.
  
- **Interactive Selection**: Users can select the company for which they want to forecast stock prices using a dropdown menu.

- **Prediction**: The app predicts future stock prices based on the selected dataset and the specified number of years for prediction.

- **Visualization**: The app provides visualizations of both raw historical data and forecasted stock prices using Plotly.

## Algorithms Used

### Convolutional Neural Networks (CNNs)

- **Model Architecture**: The forecasting model is built using TensorFlow's Keras API. It consists of a Convolutional Neural Network (CNN) followed by dense layers.
  
- **Sequence Generation**: Historical stock price data is transformed into sequences of data points, which are used as input to the CNN model. This sequence generation is crucial for time-series forecasting tasks.

- **Training**: The model is trained using historical stock price data. It minimizes the Mean Squared Error (MSE) loss function and utilizes the Adam optimizer.

- **Prediction**: After training, the model is used to predict future stock prices based on the last available data points.

## Usage

1. Install the required libraries listed in `requirements.txt`.
2. Run the Streamlit app by executing the Python script.
3. Select the company for which you want to forecast stock prices.
4. Choose the number of years for prediction using the slider.
5. Explore the raw data and forecasted prices visually.

## Dependencies

- Streamlit
- yfinance
- NumPy
- Pandas
- TensorFlow
- Plotly

## Contributors

- Abhigyan Kashyap
- Bhavya Malhotra
- Jayatri Banerjee
