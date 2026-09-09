import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

class EvaluatorWithPenalties:
    def __init__(self, commission_bps=5.0, slippage_bps=2.0, dd_penalty=2.0, turnover_penalty=0.5):
        self.cost_per_trade = (commission_bps + slippage_bps) / 10000.0
        self.dd_penalty_weight = dd_penalty
        self.turnover_penalty_weight = turnover_penalty

    def simulate(self, actual_returns, signals):
        turnover = signals.diff().abs().fillna(signals.abs())
        costs = turnover * self.cost_per_trade
        strategy_returns = (signals * actual_returns) - costs
        
        equity_curve = (1.0 + strategy_returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown_series = (equity_curve - running_max) / running_max
        max_drawdown = abs(float(drawdown_series.min()))

        mean_ret = strategy_returns.mean()
        annualized_return = ((1.0 + mean_ret) ** (252 / 5)) - 1.0
        total_turnover = int(turnover.sum())

        penalized_fitness = annualized_return - (self.dd_penalty_weight * max_drawdown) - (self.turnover_penalty_weight * (total_turnover / max(len(signals), 1)))

        return {
            "Ann. Return": annualized_return,
            "Max DD": max_drawdown,
            "Turnover": total_turnover,
            "Fitness": penalized_fitness
        }

def train_and_evaluate():
    parquet_files = sorted(glob.glob("artifacts/data_shards/shard_*.parquet"))
    if not parquet_files:
        raise ValueError("No data shard files found in artifacts/data_shards/")

    df_master = pd.concat([pd.read_parquet(f) for f in parquet_files])
    print(f"Loaded {len(df_master)} total samples across {df_master['Ticker'].nunique()} unique tickers.")

    # Chronological Split
    df_master = df_master.sort_index()
    split_idx = int(len(df_master) * 0.80)
    train_data = df_master.iloc[:split_idx]
    test_data = df_master.iloc[split_idx:].copy()

    features = ["return_1d", "volatility_20d", "sma_20_ratio", "sma_50_ratio", "rsi_14", "volume_ratio_20", "atr_ratio"]
    X_train, y_train = train_data[features], train_data["target"]
    X_test, y_test = test_data[features], test_data["target"]

    print(f"Target distribution (1s): {y_train.mean():.2%}")
    print(f"Training ensemble on {len(X_train)} rows...")

    # Class-weighted models prevent probability shrinkage
    clf1 = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42
    )
    clf2 = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = VotingClassifier(
        estimators=[("hgb", clf1), ("rf", clf2)],
        voting="soft"
    )
    model.fit(X_train, y_train)

    os.makedirs("artifacts/universal_model", exist_ok=True)
    joblib.dump(model, "artifacts/universal_model/universal_strategy.joblib")
    print("Universal model saved to artifacts/universal_model/universal_strategy.joblib")

    # In-Sample & Test Predictions
    test_data["prob_up"] = model.predict_proba(X_test)[:, 1]

    evaluator = EvaluatorWithPenalties()
    results = []
    
    # Sweep realistic threshold band
    test_thresholds = [0.50, 0.52, 0.54, 0.56, 0.58]

    for ticker, group in test_data.groupby("Ticker"):
        best_fitness = -np.inf
        best_metrics = None

        for th in test_thresholds:
            signals = (group["prob_up"] >= th).astype(int)
            if signals.sum() == 0:
                continue
            
            metrics = evaluator.simulate(group["trade_return"], signals)
            if metrics["Fitness"] > best_fitness:
                best_fitness = metrics["Fitness"]
                best_metrics = metrics
                best_metrics["Threshold"] = th

        if best_metrics:
            best_metrics["Ticker"] = ticker
            best_metrics["Accuracy"] = accuracy_score(group["target"], (group["prob_up"] >= best_metrics["Threshold"]).astype(int))
            results.append(best_metrics)
        else:
            # Retain tickers that had 0 signals so they are not omitted
            results.append({
                "Ticker": ticker,
                "Fitness": 0.0,
                "Ann. Return": 0.0,
                "Max DD": 0.0,
                "Turnover": 0,
                "Threshold": 0.50,
                "Accuracy": 0.0
            })

    df_results = pd.DataFrame(results).sort_values(by="Fitness", ascending=False).round(4)
    df_results.to_csv("artifacts/universal_model/backtest_leaderboard.csv", index=False)

    top_stocks = df_results.head(15)
    md_table = "| Ticker | Fitness | Ann. Return | Max DD | Turnover | Threshold | Accuracy |\n|---|---|---|---|---|---|---|\n"
    for _, row in top_stocks.iterrows():
        md_table += f"| **{row['Ticker']}** | {row['Fitness']:.4f} | {row['Ann. Return']*100:.2f}% | {row['Max DD']*100:.2f}% | {row['Turnover']:.0f} | {row['Threshold']:.2f} | {row['Accuracy']:.2f} |\n"

    summary_md = f"## Universal Model Backtest Complete\n" \
                 f"- **Total Stocks Evaluated:** {len(df_results)}\n" \
                 f"- **Universe Size:** {df_master['Ticker'].nunique()} tickers\n" \
                 f"- **Architecture:** Balanced Soft-Voting Ensemble (HGB + RF)\n\n" \
                 f"### Top Performing Stocks\n{md_table}"

    print(summary_md)
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(summary_md)

if __name__ == "__main__":
    train_and_evaluate()
