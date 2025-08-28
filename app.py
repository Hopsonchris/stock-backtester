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

# Section 2: Live Quotes
st.header("💹 Live Stock Quotes Section")
live_symbol = st.text_input("Enter stock symbol for live quote (e.g., AAPL)")

if st.button("Get Live Quote"):
    if live_symbol:
        try:
            ticker = yf.Ticker(live_symbol)

            # Fetch live price
            quote = ticker.info.get("currentPrice", "N/A")
            st.success(f"Current price for {live_symbol}: ${quote}")

            # Fetch recent historical data
            df = ticker.history(period="3mo", interval="1d")
            df.dropna(inplace=True)

            # ---------------- RSI 14 ---------------- #
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            df["RSI_14"] = 100 - (100 / (1 + rs))

            # ---------------- CCI 20 ---------------- #
            tp = (df["High"] + df["Low"] + df["Close"]) / 3
            sma_tp = tp.rolling(20).mean()
            mean_dev = (tp - sma_tp).abs().rolling(20).mean()
            df["CCI_20"] = (tp - sma_tp) / (0.015 * mean_dev)

            # ---------------- Plot: Price + Volume ---------------- #
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            ))
            fig_price.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                marker_color="lightblue",
                yaxis="y2"
            ))

            fig_price.update_layout(
                title=f"{live_symbol} Price & Volume",
                xaxis_title="Date",
                yaxis_title="Price",
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # ---------------- Plot: RSI ---------------- #
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index,
                y=df["RSI_14"],
                mode="lines",
                name="RSI 14",
                line=dict(color="orange")
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")

            fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)

            # ---------------- Plot: CCI ---------------- #
            fig_cci = go.Figure()
            fig_cci.add_trace(go.Scatter(
                x=df.index,
                y=df["CCI_20"],
                mode="lines",
                name="CCI 20",
                line=dict(color="purple")
            ))
            fig_cci.add_hline(y=100, line_dash="dash", line_color="red")
            fig_cci.add_hline(y=-100, line_dash="dash", line_color="green")

            fig_cci.update_layout(title="CCI (20)", yaxis_title="CCI")
            st.plotly_chart(fig_cci, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching live quote: {e}")
    else:
        st.warning("Please enter a stock symbol.")

# Section 3: Stock and Fund Screener (Added below Live Quotes)
st.header("🔍 Stock and Fund Screener")
st.write("Screen stocks and funds (e.g., ETFs) based on fundamental criteria. Enter a list of symbols and apply filters.")

screener_symbols = st.text_input("Enter symbols to screen (comma-separated, e.g., AAPL,MSFT,SPY,VWO)", "")

col1, col2, col3 = st.columns(3)

with col1:
    min_market_cap = st.number_input("Min Market Cap (in billions)", min_value=0.0, value=0.0, step=0.1) * 1e9
    max_pe = st.number_input("Max P/E Ratio", min_value=0.0, value=0.0, step=0.1)

with col2:
    min_roe = st.number_input("Min ROE (%)", min_value=0.0, value=0.0, step=0.1) / 100
    min_div_yield = st.number_input("Min Dividend Yield (%)", min_value=0.0, value=0.0, step=0.01) / 100

with col3:
    max_beta = st.number_input("Max Beta", min_value=0.0, value=0.0, step=0.1)
    sector = st.text_input("Sector (exact match, e.g., Technology)", "")

if st.button("Run Screener"):
    if screener_symbols:
        symbols_list = [s.strip().upper() for s in screener_symbols.split(",")]
        with st.spinner("Fetching data and screening... this may take a moment."):
            df = fetch_fundamental_metrics(symbols_list)
            
            # Convert relevant columns to numeric, handling 'N/A'
            numeric_cols = ["Market Cap", "P/E Ratio", "ROE", "Dividend Yield", "Beta"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Apply filters only if values are set (non-zero for mins/maxes)
            filtered_df = df.copy()
            if min_market_cap > 0:
                filtered_df = filtered_df[filtered_df["Market Cap"] >= min_market_cap]
            if max_pe > 0:
                filtered_df = filtered_df[filtered_df["P/E Ratio"] <= max_pe]
            if min_roe > 0:
                filtered_df = filtered_df[filtered_df["ROE"] >= min_roe]
            if min_div_yield > 0:
                filtered_df = filtered_df[filtered_df["Dividend Yield"] >= min_div_yield]
            if max_beta > 0:
                filtered_df = filtered_df[filtered_df["Beta"] <= max_beta]
            if sector:
                filtered_df = filtered_df[filtered_df["Sector"].str.lower() == sector.lower()]  # Case-insensitive match
            
            # Format for display (similar to Analyze Stocks)
            display_df = filtered_df.copy()
            for col in ["Market Cap"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            for col in ["P/E Ratio", "Forward P/E", "EPS", "ROE", "Debt/Equity", "P/B Ratio", "PEG Ratio", "Beta", "52 Week High", "52 Week Low"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            if "Dividend Yield" in display_df.columns:
                display_df["Dividend Yield"] = display_df["Dividend Yield"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
            
            st.subheader("Screened Results")
            st.dataframe(display_df, use_container_width=True)
            
            if filtered_df.empty:
                st.info("No symbols match the criteria.")
    else:
        st.warning("Please enter at least one symbol.")