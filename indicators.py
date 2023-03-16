def indiOut(df):
    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    import streamlit as st

    df = pd.DataFrame(df, columns = ['Close'])
    
    # create 20 days simple moving average column
    df['20_SMA'] = df['Close'].rolling(window = 20, min_periods = 1).mean()
    # create 50 days simple moving average column
    df['50_SMA'] = df['Close'].rolling(window = 50, min_periods = 1).mean()

    df['Signal'] = 0.0
    df['Signal'] = np.where(df['20_SMA'] > df['50_SMA'], 1.0, 0.0)

    df['Position'] = df['Signal'].diff()

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df.index , y=df['Close'],
                        mode='lines', marker_color='white',
                        name='Close', opacity=0.6))
    fig.add_trace(go.Scatter(x=df[df['Position'] == 1].index, y=df['20_SMA'][df['Position'] == 1],
                        mode='markers', name='Buy', marker_color='green', marker_symbol='diamond', marker_size=7))
    fig.add_trace(go.Scatter(x=df[df['Position'] == -1].index, y=df['20_SMA'][df['Position'] == -1],
                        mode='markers', name='Buy', marker_color='red', marker_symbol='diamond', marker_size=7))
    fig.update_layout(title_text='Predicted Close Price With Indicators', font_size=15)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
