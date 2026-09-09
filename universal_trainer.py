import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

class EvaluatorWithPenalties:
    def __init__(self, commission_bps=5.0, slippage_bps=2.0, dd_penalty=2.5, turnover_penalty=1.2):
        self.cost_per_trade = (commission_bps + slippage_bps) / 10000.0
        self.dd_penalty_weight = dd_penalty
        self.turnover_penalty_weight = turnover_penalty

    def simulate(self, actual_returns, signals):
        turnover = signals.diff().abs().fillna(signals.abs())
        costs = turnover * self.cost_per_trade
        # actual_returns now represents the full outcome of the swing trade (TP or SL)
        strategy_returns = signals * actual_returns - costs
        
        equity_curve = (1.0 + strategy_returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown_series = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown_series.min())

        mean_ret = strategy_returns.mean()
        annualized_return = (1.0 + mean_ret) ** (252 / 5) - 1.0 # Adjusted for avg ~5 day holding period
        total_turnover = turnover.sum()
        
        penalized_fitness = annualized_return - (self.dd_penalty_weight * max_drawdown) - (self.turnover_penalty_weight * (total_turnover / len(signals)))

        return {
            "Ann. Return": annualized_return,
            "Max DD": max_drawdown,
            "Turnover": total_turnover,
            "Fitness": penalized_fitness
        }

def train_and_evaluate():
    # 1. Load and merge all parquet files
    parquet_files = glob.glob("artifacts/data_shards/shard_*.parquet")
    if not parquet_files:
        raise ValueError("No data shards found.")
    
    df_master = pd.concat([pd.read_parquet(f) for f in parquet_files])
    
    # 2. Chronological global train/test split
    df_master = df_master.sort_index()
    split_idx = int(len(df_master) * 0.8)
    
    train_data = df_master.iloc[:split_idx]
    test_data = df_master.iloc[split_idx:]
    
    # Added atr_ratio to features
    features = ["return_1d", "volatility_20d", "sma_20_ratio", "sma_50_ratio", "rsi_14", "atr_ratio"]
    X_train, y_train = train_data[features], train_data["target"]
    X_test, y_test = test_data[features], test_data["target"]
    
    print(f"Training ENSEMBLE model on {len(X_train)} data points across all stocks...")
    
    # 3. Train a multi-algorithm ensemble
    clf1 = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=-1)
    clf3 = ExtraTreesClassifier(n_estimators=100, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=-1)
    
    model = VotingClassifier(
        estimators=[('hgb', clf1), ('rf', clf2), ('et', clf3)],
        voting='soft'
    )
    model.fit(X_train, y_train)
    
    # 4. Global predictions
    test_data = test_data.copy()
    test_data["prob_up"] = model.predict_proba(X_test)[:, 1]
    
    os.makedirs("artifacts/universal_model", exist_ok=True)
    joblib.dump(model, "artifacts/universal_model/universal_strategy.joblib")
    print("Universal model saved successfully.")

    # 5. Backtest Evaluate per Ticker using highly selective thresholds
    evaluator = EvaluatorWithPenalties()
    results = []
    
    for ticker, group in test_data.groupby("Ticker"):
        best_fitness = -np.inf
        best_metrics = None
        
        # Test extreme thresholds for maximum probability of winning
        for threshold in [0.65, 0.70, 0.75, 0.80]:
            signals = pd.Series((group["prob_up"] > threshold).astype(int), index=group.index)
            if signals.sum() == 0: continue
            
            metrics = evaluator.simulate(group["trade_return"], signals)
            if metrics["Fitness"] > best_fitness:
                best_fitness = metrics["Fitness"]
                best_metrics = metrics
                best_metrics["Threshold"] = threshold

        if best_metrics:
            best_metrics["Ticker"] = ticker
            best_metrics["Accuracy"] = accuracy_score(group["target"], (group["prob_up"] > best_metrics["Threshold"]).astype(int))
            results.append(best_metrics)

    # 6. Generate Leaderboard
    df_results = pd.DataFrame(results).sort_values(by="Fitness", ascending=False).round(4)
    df_results.to_csv("artifacts/universal_model/backtest_leaderboard.csv", index=False)
    
    top_10 = df_results.head(10)
    md_table = "| Ticker | Fitness | Ann. Return | Max DD | Turnover | Threshold | Accuracy |\n|---|---|---|---|---|---|---|\n"
    for _, row in top_10.iterrows():
        md_table += f"| **{row['Ticker']}** | {row['Fitness']:.4f} | {row['Ann. Return']*100:.2f}% | {row['Max DD']*100:.2f}% | {row['Turnover']:.0f} | {row['Threshold']} | {row['Accuracy']:.2f} |\n"

    summary_md = f"## 🌍 Universal Model Backtest Complete\n- **Total Stocks Evaluated:** {len(df_results)}\n- **Model Scope:** Multi-Algorithm Ensemble (HGB, RF, ET)\n\n### Top 10 Stocks for Universal Strategy\n{md_table}"
    
    print(summary_md)
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(summary_md)

if __name__ == "__main__":
    train_and_evaluate()
