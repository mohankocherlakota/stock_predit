import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.title('Stock Price Prediction')

ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL)', 'AAPL')

if ticker:
    @st.cache_data
    def load_data(ticker):
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=90)
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(ticker)
    data_load_state.text('Loading data...done!')

    st.subheader('Raw data')
    st.write(data.tail())

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    X = data[['Close', 'Volume', 'High', 'Low']]
    y = data['Returns']

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    algorithms = ['Decision Tree', 'ARIMA', 'LSTM', 'Gradient Boosting']
    results = {}

    for algorithm in algorithms:
        cv_scores = []

        if algorithm == 'Decision Tree':
            model = DecisionTreeRegressor(random_state=42, max_depth=5)  # Added max_depth to reduce overfitting
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores.append(mean_squared_error(y_test, y_pred))
            model.fit(X, y)
            y_pred = model.predict(X)
            future_X = X.iloc[-1:].values.repeat(30, axis=0)
            future_pred = model.predict(future_X)

        elif algorithm == 'ARIMA':
            for train_index, test_index in tscv.split(y):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model = ARIMA(y_train, order=(1,1,1))
                model_fit = model.fit()
                y_pred = model_fit.forecast(steps=len(y_test))
                cv_scores.append(mean_squared_error(y_test, y_pred))
            model = ARIMA(y, order=(1,1,1))
            model_fit = model.fit()
            y_pred = model_fit.fittedvalues
            future_pred = model_fit.forecast(steps=30)

        elif algorithm == 'LSTM':
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            for train_index, test_index in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True),
                    Dropout(0.2),
                    LSTM(50, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, 
                          epochs=100, batch_size=32, validation_split=0.2, 
                          callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).flatten()
                cv_scores.append(mean_squared_error(y_test, y_pred))
            
            model.fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1), y, 
                      epochs=100, batch_size=32, validation_split=0.2, 
                      callbacks=[early_stopping], verbose=0)
            y_pred = model.predict(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)).flatten()
            future_X = X.iloc[-1:].values.repeat(30, axis=0)
            future_X_scaled = scaler.transform(future_X)
            future_pred = model.predict(future_X_scaled.reshape(30, X_scaled.shape[1], 1)).flatten()

        else:  # Gradient Boosting
            model = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores.append(mean_squared_error(y_test, y_pred))
            model.fit(X, y)
            y_pred = model.predict(X)
            future_X = X.iloc[-1:].values.repeat(30, axis=0)
            future_pred = model.predict(future_X)

        mse = np.mean(cv_scores)
        results[algorithm] = {'mse': mse, 'y_pred': y_pred, 'future_pred': future_pred}

        st.subheader(f'{algorithm} - Historical Data and 30-Day Forecast')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, y, label='Historical Returns', alpha=0.7)
        ax.plot(data.index, y_pred, label='Fitted Values', alpha=0.7)
        ax.plot(future_dates, future_pred, label='30-Day Forecast', color='red')
        ax.axvline(x=last_date, color='black', linestyle='--', label='Forecast Start')
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.set_title(f'{ticker} Stock Returns - Historical and Forecast using {algorithm}')
        st.pyplot(fig)

    st.subheader('Model Comparison')
    comparison_df = pd.DataFrame({alg: results[alg]['mse'] for alg in algorithms}, index=['Mean Squared Error']).T
    comparison_df = comparison_df.sort_values('Mean Squared Error')
    st.write(comparison_df)

    best_model = comparison_df.index[0]
    st.write(f"The best performing model is: {best_model}")
