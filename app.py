import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS for Premium Look
# ─────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .signal-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .buy { background-color: #00c853; color: white; }
    .sell { background-color: #ff1744; color: white; }
    .hold { background-color: #ffab00; color: white; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar Configuration
# ─────────────────────────────────────────────────────────────
st.sidebar.title("📈 Stock Agent")
st.sidebar.markdown("---")

TICKER_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOG": "GOOG",
    "TSLA": "TSLA",
}

selected_symbol = st.sidebar.selectbox("Select Stock Symbol", list(TICKER_MAP.keys()))
yahoo_ticker = TICKER_MAP[selected_symbol]

st.sidebar.markdown("---")
st.sidebar.info("""
    This dashboard uses a **3-Layer Stacked LSTM** neural network to predict next-day closing prices based on the past 60 days of OHLCV data.
""")

# ─────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_prediction(symbol):
    try:
        response = requests.get(f"http://localhost:5001/predict?symbol={symbol}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Failed: {str(e)}"}

@st.cache_data(ttl=600)
def fetch_historical_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ─────────────────────────────────────────────────────────────
# Main Dashboard
# ─────────────────────────────────────────────────────────────
st.title(f"📊 {selected_symbol} Prediction Dashboard")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Fetch Prediction Data
with st.spinner("Fetching predictions from AI model..."):
    pred_data = get_prediction(selected_symbol)

if "error" in pred_data:
    st.error(pred_data["error"])
    st.info("Make sure the FastAPI server is running on port 5001!")
else:
    # Top Row: Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"₹{pred_data['current_price']:,}")
    
    with col2:
        diff = pred_data['predicted_price'] - pred_data['current_price']
        st.metric("Predicted Next Close", f"₹{pred_data['predicted_price']:,}", f"{pred_data['pct_change']}%")
        
    with col3:
        st.metric("10-day MA", f"₹{pred_data['short_ma']:,}")
        
    with col4:
        st.metric("30-day MA", f"₹{pred_data['long_ma']:,}")

    # Signal Indicator
    signal = pred_data['signal']
    signal_class = "buy" if signal == "BUY" else "sell" if signal == "SELL" else "hold"
    st.markdown(f'<div class="signal-card {signal_class}">TRAFFIC SIGNAL: {signal}</div>', unsafe_allow_html=True)

    # Historical Chart
    st.subheader("Historical Performance & Moving Averages")
    hist_df = fetch_historical_data(yahoo_ticker)
    
    if not hist_df.empty:
        fig = go.Figure()
        
        # Actual Close
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['Close'],
            name='Actual Close',
            line=dict(color='#448aff', width=2)
        ))
        
        # Moving Averages
        hist_df['MA10'] = hist_df['Close'].rolling(window=10).mean()
        hist_df['MA30'] = hist_df['Close'].rolling(window=30).mean()
        
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['MA10'],
            name='10-day MA',
            line=dict(color='#ffab00', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['MA30'],
            name='30-day MA',
            line=dict(color='#00e676', width=1, dash='dot')
        ))
        
        # Add Prediction Point
        next_day = hist_df.index[-1] + timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[next_day], y=[pred_data['predicted_price']],
            name='LSTM Prediction',
            mode='markers',
            marker=dict(color='#ff1744', size=12, symbol='star')
        ))

        fig.update_layout(
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Price (INR ₹)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch historical chart data.")


