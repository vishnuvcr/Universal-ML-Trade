import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_and_prepare_data(ticker: str, period: str = "4y") -> pd.DataFrame:
    yf_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    df = yf.download(yf_ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    
    if df.empty or len(df) < 250:
        return pd.DataFrame()
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Core Technical Features
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["sma_20_ratio"] = (df["Close"] / df["Close"].rolling(20).mean()) - 1.0
    df["sma_50_ratio"] = (df["Close"] / df["Close"].rolling(50).mean()) - 1.0
    
    # RSI (14-period)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    
    # Volume Dynamics
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio_20"] = df["Volume"] / (df["vol_sma_20"] + 1e-9)

    # 2. Dynamic ATR Calculation
    df["High-Low"] = df["High"] - df["Low"]
    df["High-PClose"] = (df["High"] - df["Close"].shift(1)).abs()
    df["Low-PClose"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["High-Low", "High-PClose", "Low-PClose"]].max(axis=1)
    df["ATR_14"] = df["TR"].rolling(14).mean()
    df["atr_ratio"] = df["ATR_14"] / df["Close"]

    # 3. Dynamic Swing Order Levels (1.5:1 Reward-to-Risk)
    df["SL_Price"] = df["Close"] - (df["ATR_14"] * 1.0)
    df["TP_Price"] = df["Close"] + (df["ATR_14"] * 1.5)

    # 4. Target Generation (5-Day Holding Period)
    future_high = df["High"].shift(-1).rolling(5).max()
    future_low = df["Low"].shift(-1).rolling(5).min()

    # Target = 1 if Take Profit hit BEFORE Stop Loss
    df["target"] = ((future_high >= df["TP_Price"]) & (future_low > df["SL_Price"])).astype(int)

    # Simulated outcome return for evaluator
    df["trade_return"] = np.where(
        df["target"] == 1,
        (df["TP_Price"] - df["Close"]) / df["Close"],
        (df["SL_Price"] - df["Close"]) / df["Close"]
    )

    df.drop(columns=["High-Low", "High-PClose", "Low-PClose", "TR", "vol_sma_20"], inplace=True)
    df.dropna(inplace=True)
    df["Ticker"] = ticker.replace(".NS", "")
    return df

def run_data_build():
    shard_index = int(os.getenv("SHARD_INDEX", 0))
    total_shards = int(os.getenv("TOTAL_SHARDS", 1))

    if not os.path.exists("tickers.txt"):
        raise FileNotFoundError("tickers.txt file not found.")

    with open("tickers.txt", "r") as f:
        tickers = sorted([line.strip() for line in f if line.strip() and not line.startswith("#")])

    my_tickers = tickers[shard_index::total_shards]
    print(f"Runner {shard_index + 1}/{total_shards} downloading {len(my_tickers)} tickers...")

    all_data = []
    for i, ticker in enumerate(my_tickers):
        print(f"[{i + 1}/{len(my_tickers)}] Fetching {ticker}...")
        df = fetch_and_prepare_data(ticker)
        if not df.empty:
            all_data.append(df)
        time.sleep(1.0)

    if not all_data:
        print("No valid data collected in this shard.")
        return

    df_shard = pd.concat(all_data)
    os.makedirs("artifacts/data_shards", exist_ok=True)
    df_shard.to_parquet(f"artifacts/data_shards/shard_{shard_index}.parquet")
    print(f"Shard {shard_index} saved {len(df_shard)} rows across {df_shard['Ticker'].nunique()} stocks.")

if __name__ == "__main__":
    run_data_build()
