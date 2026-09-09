import os
import time
import pandas as pd
import yfinance as yf

def fetch_and_prepare_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    yf_ticker = f"{ticker}.NS"
    df = yf.download(yf_ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty or len(df) < 500:
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Feature Engineering (Done per stock to avoid cross-contamination)
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["sma_20_ratio"] = df["Close"] / df["Close"].rolling(20).mean() - 1.0
    df["sma_50_ratio"] = df["Close"] / df["Close"].rolling(50).mean() - 1.0

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Shift target and drop NaNs created by rolling windows
    df["target"] = (df["return_1d"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    
    # Tag the data with the ticker symbol
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

    # Combine all stocks in this shard into one DataFrame
    df_shard = pd.concat(all_data)
    
    os.makedirs("artifacts/data_shards", exist_ok=True)
    df_shard.to_parquet(f"artifacts/data_shards/shard_{shard_index}.parquet")
    print(f"Shard {shard_index} saved {len(df_shard)} rows of data.")

if __name__ == "__main__":
    run_data_build()
