
import pandas as pd
import numpy as np

def add_technical_indicators(data):
    """
    Adds SMA, EMA, RSI, MACD, and Bollinger Bands to each ticker in a MultiIndex DataFrame.
    Assumes data.columns = MultiIndex [Ticker, ['Open', 'High', 'Low', 'Close', 'Volume']]
    """

    for ticker in data.columns.levels[0]:
        close = data[ticker]['Close']
        high = data[ticker]['High']
        low = data[ticker]['Low']

        # SMA & EMA
        data[(ticker, 'SMA_20')] = close.rolling(window=20).mean()
        data[(ticker, 'EMA_20')] = close.ewm(span=20, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data[(ticker, 'RSI_14')] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        data[(ticker, 'MACD')] = macd
        data[(ticker, 'MACD_signal')] = signal

        # Bollinger Bands
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        data[(ticker, 'BB_upper')] = sma + (2 * std)
        data[(ticker, 'BB_middle')] = sma
        data[(ticker, 'BB_lower')] = sma - (2 * std)

    return data.sort_index(axis=1)


import plotly.graph_objects as go

def plot_technical_indicators(data, ticker):
    """
    Plots Close price with SMA, EMA, and Bollinger Bands using Plotly.
    Assumes the DataFrame includes those technical indicators.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index, y=data[(ticker, 'Close')], mode='lines', name='Close'
    ))

    fig.add_trace(go.Scatter(
        x=data.index, y=data[(ticker, 'SMA_20')], mode='lines', name='SMA 20'
    ))

    fig.add_trace(go.Scatter(
        x=data.index, y=data[(ticker, 'EMA_20')], mode='lines', name='EMA 20'
    ))

    fig.add_trace(go.Scatter(
        x=data.index, y=data[(ticker, 'BB_upper')], mode='lines', name='BB Upper', line=dict(dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data[(ticker, 'BB_middle')], mode='lines', name='BB Middle', line=dict(dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data[(ticker, 'BB_lower')], mode='lines', name='BB Lower', line=dict(dash='dot')
    ))

    fig.update_layout(
        title=f'Technical Indicators for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()
