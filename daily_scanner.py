import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

FEATURES = ["return_1d", "volatility_20d", "sma_20_ratio", "sma_50_ratio", "rsi_14", "volume_ratio_20", "atr_ratio"]

def generate_daily_signals(model_path="artifacts/universal_model/universal_strategy.joblib", min_prob=0.52):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)

    with open("tickers.txt", "r") as f:
        tickers = sorted([line.strip() for line in f if line.strip() and not line.startswith("#")])

    signals = []
    print(f"Scanning {len(tickers)} stocks for today's trade setups...")

    for ticker in tickers:
        yf_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        try:
            df = yf.download(yf_ticker, period="90d", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Reconstruct Features
            df["return_1d"] = df["Close"].pct_change()
            df["volatility_20d"] = df["return_1d"].rolling(20).std()
            df["sma_20_ratio"] = (df["Close"] / df["Close"].rolling(20).mean()) - 1.0
            df["sma_50_ratio"] = (df["Close"] / df["Close"].rolling(50).mean()) - 1.0
            
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
            
            df["vol_sma_20"] = df["Volume"].rolling(20).mean()
            df["volume_ratio_20"] = df["Volume"] / (df["vol_sma_20"] + 1e-9)

            df["High-Low"] = df["High"] - df["Low"]
            df["High-PClose"] = (df["High"] - df["Close"].shift(1)).abs()
            df["Low-PClose"] = (df["Low"] - df["Close"].shift(1)).abs()
            df["TR"] = df[["High-Low", "High-PClose", "Low-PClose"]].max(axis=1)
            df["ATR_14"] = df["TR"].rolling(14).mean()
            df["atr_ratio"] = df["ATR_14"] / df["Close"]

            df.dropna(inplace=True)
            if df.empty:
                continue

            # Latest bar
            latest_row = df.iloc[-1:]
            X_curr = latest_row[FEATURES]
            
            prob = float(model.predict_proba(X_curr)[0, 1])

            close_val = float(latest_row["Close"].iloc[0])
            atr_val = float(latest_row["ATR_14"].iloc[0])

            sl_price = round(close_val - (1.0 * atr_val), 2)
            tp_price = round(close_val + (1.5 * atr_val), 2)

            if prob >= min_prob:
                signals.append({
                    "Ticker": ticker.replace(".NS", ""),
                    "Close": round(close_val, 2),
                    "Win_Prob": round(prob, 3),
                    "Stop_Loss": sl_price,
                    "Take_Profit": tp_price,
                    "Reward_Risk": "1.5 : 1"
                })
        except Exception as e:
            continue

    df_signals = pd.DataFrame(signals)
    if not df_signals.empty:
        df_signals = df_signals.sort_values(by="Win_Prob", ascending=False)
        os.makedirs("artifacts/daily_signals", exist_ok=True)
        df_signals.to_csv("artifacts/daily_signals/today_signals.csv", index=False)

    # Markdown Summary for GitHub Actions
    md = f"## Daily Trade Signals ({pd.Timestamp.now().strftime('%Y-%m-%d')})\n"
    if df_signals.empty:
        md += "No stocks met the minimum probability threshold today.\n"
    else:
        md += f"Found **{len(df_signals)}** actionable swing trade setups:\n\n"
        md += "| Ticker | Close | Win Prob | Stop Loss | Take Profit | R:R |\n|---|---|---|---|---|---|\n"
        for _, r in df_signals.iterrows():
            md += f"| **{r['Ticker']}** | ₹{r['Close']} | {r['Win_Prob']*100:.1f}% | ₹{r['Stop_Loss']} | ₹{r['Take_Profit']} | {r['Reward_Risk']} |\n"

    print(md)
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(md)

if __name__ == "__main__":
    generate_daily_signals()
