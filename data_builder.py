import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_and_prepare_data(ticker: str, period: str = "4y") -> pd.DataFrame:
    yf_ticker = f"{ticker}.NS"
    df = yf.download(yf_ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty or len(df) < 500:
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Base Feature Engineering
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["sma_20_ratio"] = df["Close"] / df["Close"].rolling(20).mean() - 1.0
    df["sma_50_ratio"] = df["Close"] / df["Close"].rolling(50).mean() - 1.0

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 2. ATR Calculation for Dynamic SL/TP
    df["High-Low"] = df["High"] - df["Low"]
    df["High-PClose"] = abs(df["High"] - df["Close"].shift(1))
    df["Low-PClose"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["High-Low", "High-PClose", "Low-PClose"]].max(axis=1)
    df["ATR_14"] = df["TR"].rolling(14).mean()
    
    # New Feature: Normalized ATR to help the model gauge relative volatility
    df["atr_ratio"] = df["ATR_14"] / df["Close"]

    # 3. Define strict 1:2 Risk/Reward levels
    df["SL_Price"] = df["Close"] - (df["ATR_14"] * 1.5)
    df["TP_Price"] = df["Close"] + (df["ATR_14"] * 3.0)

    # 4. Target Engineering: Forward-looking 5-day window for swing trading
    future_high = df["High"].shift(-1).rolling(5).max()
    future_low = df["Low"].shift(-1).rolling(5).min()

    # Target: 1 if TP hit BEFORE SL within 5 days, 0 otherwise
    df["target"] = ((future_high >= df["TP_Price"]) & (future_low > df["SL_Price"])).astype(int)
    
    # Calculate simulated trade return for the evaluator
    df["trade_return"] = np.where(
        df["target"] == 1, 
        (df["TP_Price"] - df["Close"]) / df["Close"], 
        (df["SL_Price"] - df["Close"]) / df["Close"]
    )
    
    # Cleanup
    df.drop(columns=["High-Low", "High-PClose", "Low-PClose", "TR"], inplace=True)
    df.dropna(inplace=True)
    df["Ticker"] = ticker
    
    return df

def run_data_build():
    shard_index = int(os.getenv("SHARD_INDEX", 0))
    total_shards = int(os.getenv("TOTAL_SHARDS", 1))

    with open("tickers.txt", "r") as f:
        tickers = sorted([line.strip() for line in f if line.strip() and not line.startswith("#")])
    
    my_tickers = tickers[shard_index::total_shards]
    print(f"Runner {shard_index+1}/{total_shards} downloading data for {len(my_tickers)} tickers...")

    all_data = []
    for i, ticker in enumerate(my_tickers):
        print(f"[{i+1}/{len(my_tickers)}] Fetching {ticker}...")
        df = fetch_and_prepare_data(ticker)
        if not df.empty:
            all_data.append(df)
        time.sleep(2.5) # yfinance rate limit protection

    if not all_data:
        print("No data collected in this shard.")
        return

    df_shard = pd.concat(all_data)
    
    os.makedirs("artifacts/data_shards", exist_ok=True)
    df_shard.to_parquet(f"artifacts/data_shards/shard_{shard_index}.parquet")
    print(f"Shard {shard_index} saved {len(df_shard)} rows of data.")

if __name__ == "__main__":
    run_data_build()
