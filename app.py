# app.py
# Stock Analysis and Backtesting App
# Run with: streamlit run app.py

import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ------------------------------
# Function to fetch historical data
# ------------------------------
def fetch_historical_data(symbols, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                st.warning(f"No data found for {symbol}")
                continue
            # Use "Adj Close" for calculations as it accounts for dividends and splits
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            data[symbol] = df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
    return data

# ------------------------------
# Fundamental Analysis Function
# ------------------------------
def fetch_fundamental_metrics(symbols):
    """Fetches key fundamental metrics for a list of stock symbols."""
    results = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Collect desired metrics, providing 'N/A' for missing values
            metrics = {
                "Symbol": symbol,
                "Company Name": info.get("shortName", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "P/E Ratio": info.get("trailingPE", "N/A"),
                "Forward P/E": info.get("forwardPE", "N/A"),
                "EPS": info.get("trailingEps", "N/A"),
                "ROE": info.get("returnOnEquity", "N/A"),
                "Debt/Equity": info.get("debtToEquity", "N/A"),
                "P/B Ratio": info.get("priceToBook", "N/A"),
                "PEG Ratio": info.get("pegRatio", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A"),
                "Beta": info.get("beta", "N/A"),
                "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
                "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            }
            results.append(metrics)
        except Exception as e:
            results.append({"Symbol": symbol, "Error": str(e)})
    return pd.DataFrame(results)

# ------------------------------
# Backtest Strategies
# ------------------------------
def backtest_ma_crossover(df, short_window, long_window):
    """Generates trading signals for a moving average crossover strategy."""
    df["short_ma"] = df["Close"].rolling(window=short_window, min_periods=1).mean()
    df["long_ma"] = df["Close"].rolling(window=long_window, min_periods=1).mean()
    df["signal"] = 0
    # Generate signals
    df.loc[df.index[short_window:], "signal"] = np.where(
        df["short_ma"].iloc[short_window:] > df["long_ma"].iloc[short_window:], 1, 0
    )
    # A position is the same as the signal (1 for long, 0 for out)
    df["positions"] = df["signal"]
    return df

def backtest_buy_hold(df):
    """Simulates a buy-and-hold strategy."""
    df["positions"] = 1  # Always in a long position
    return df

# ------------------------------
# Compute performance with metrics
# ------------------------------
def compute_performance(df, initial_capital=100000, slippage=0.0005, commission=0.0005):
    """Calculates performance metrics for a backtest."""
    df["returns"] = df["Close"].pct_change()
    
    # Apply trading costs to strategy returns
    df["strategy_returns"] = df["returns"] * df["positions"].shift(1).fillna(0)
    trade_occurred = (df["positions"].diff().abs() > 0)
    df.loc[trade_occurred, "strategy_returns"] -= (slippage + commission)

    df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
    df["equity"] = initial_capital * df["cumulative_returns"]

    # Calculate metrics
    total_return = df["cumulative_returns"].iloc[-1] - 1
    max_drawdown = ((df["equity"].cummax() - df["equity"]) / df["equity"].cummax()).max()
    
    # Ensure standard deviation is not zero to avoid division errors
    if df["strategy_returns"].std() != 0:
        sharpe_ratio = (df["strategy_returns"].mean() / df["strategy_returns"].std()) * (252 ** 0.5)
    else:
        sharpe_ratio = 0
        
    positive_returns = df.loc[df["strategy_returns"] > 0, "strategy_returns"].sum()
    negative_returns = abs(df.loc[df["strategy_returns"] < 0, "strategy_returns"].sum())
    
    profit_factor = positive_returns / negative_returns if negative_returns > 0 else np.nan
        
    if len(df) > 0:
        cagr = (df["cumulative_returns"].iloc[-1]) ** (252 / len(df)) - 1
    else:
        cagr = 0
        
    mar_ratio = cagr / max_drawdown if max_drawdown > 0 else np.nan

    metrics = {
        "Total Return": f"{total_return * 100:.2f}%",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Profit Factor": f"{profit_factor:.2f}" if not pd.isna(profit_factor) else "N/A",
        "CAGR": f"{cagr * 100:.2f}%",
        "MAR Ratio": f"{mar_ratio:.2f}" if not pd.isna(mar_ratio) else "N/A",
    }
    return df, metrics

# ------------------------------
# Plotting functions
# ------------------------------
def plot_equity_curve(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["equity"], mode="lines", name="Equity Curve", line=dict(color='green')))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity ($)")
    return fig

def plot_price_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))
    if "short_ma" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["short_ma"], line=dict(color="blue", width=1.5), name="Short MA"))
    if "long_ma" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["long_ma"], line=dict(color="red", width=1.5), name="Long MA"))
    fig.update_layout(title=f"Price Chart for {symbol}", xaxis_title="Date", yaxis_title="Price ($)", xaxis_rangeslider_visible=False)
    return fig

# ------------------------------
# Streamlit App UI
# ------------------------------
st.set_page_config(page_title="Stock Analysis App", layout="wide")
st.title("📈 Stock Analysis and Backtesting App")

st.sidebar.header("⚙️ Controls")
symbols_input = st.sidebar.text_input("Enter symbols (e.g., AAPL, MSFT, GOOG)", "AAPL,GOOG")

# UPDATED: Added "Analyze Stocks" as the first option
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Analyze Stocks", "Buy and Hold", "Moving Average Crossover"]
)

# --- Logic to conditionally display inputs based on the selected analysis type ---
if analysis_type == "Analyze Stocks":
    st.sidebar.info("Fetches and displays key fundamental metrics for the selected stocks.")
else:
    # These inputs are only for backtesting
    st.sidebar.subheader("Backtest Parameters")
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*5))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    
    if analysis_type == "Moving Average Crossover":
        short_window = st.sidebar.number_input("Short MA Window", min_value=1, value=50)
        long_window = st.sidebar.number_input("Long MA Window", min_value=1, value=200)

    st.sidebar.subheader("Trading Costs")
    slippage = st.sidebar.number_input("Slippage (%)", value=0.05, step=0.01, format="%.2f") / 100
    commission = st.sidebar.number_input("Commission (%)", value=0.05, step=0.01, format="%.2f") / 100

# Central "Run" button
if st.sidebar.button("▶️ Run Analysis"):
    if symbols_input:
        symbols = [s.strip().upper() for s in symbols_input.split(",")]

        # --- Main logic block for handling different analysis types ---
        if analysis_type == "Analyze Stocks":
            st.header("📊 Fundamental Analysis")
            with st.spinner("Fetching fundamental data... this may take a moment."):
                fundamental_data = fetch_fundamental_metrics(symbols)
                # Format specific columns for better readability
                for col in ["Market Cap"]:
                    if col in fundamental_data.columns:
                        fundamental_data[col] = fundamental_data[col].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else "N/A")
                for col in ["P/E Ratio", "Forward P/E", "EPS", "ROE", "Debt/Equity", "P/B Ratio", "PEG Ratio", "Beta", "52 Week High", "52 Week Low"]:
                     if col in fundamental_data.columns:
                        fundamental_data[col] = fundamental_data[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "N/A")
                if "Dividend Yield" in fundamental_data.columns:
                    fundamental_data["Dividend Yield"] = fundamental_data["Dividend Yield"].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else "N/A")

                st.dataframe(fundamental_data, use_container_width=True)

        else: # Handle backtesting strategies
            st.header("📈 Backtest Results")
            with st.spinner("Fetching data and running backtest..."):
                data = fetch_historical_data(symbols, start_date, end_date)
                
                if not data:
                    st.error("Could not fetch data for the entered symbols. Please check the symbols and date range.")
                else:
                    for symbol, df in data.items():
                        st.subheader(f"Results for: {symbol}")

                        if analysis_type == "Buy and Hold":
                            df_strategy = backtest_buy_hold(df.copy())
                        elif analysis_type == "Moving Average Crossover":
                            df_strategy = backtest_ma_crossover(df.copy(), short_window, long_window)

                        df_perf, metrics = compute_performance(df_strategy, slippage=slippage, commission=commission)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("#### Performance Metrics")
                            st.json(metrics)
                        with col2:
                            st.plotly_chart(plot_equity_curve(df_perf), use_container_width=True)
                        
                        st.plotly_chart(plot_price_chart(df_perf, symbol), use_container_width=True)
                        st.write("---")
    else:
        st.sidebar.warning("Please enter at least one stock symbol.")