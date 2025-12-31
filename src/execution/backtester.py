import pandas as pd
import numpy as np
import os
import joblib
import sys
import matplotlib.pyplot as plt
import json
from typing import Dict

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import DataLoader
from features.engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    """
    Vectorized and Step-through Backtester for Regime-Aware Strategy.
    """
    def __init__(self, ticker="SPY"):
        self.ticker = ticker
        try:
            self.ensemble = joblib.load("models/regime_ensemble.joblib")
            self.forecaster = joblib.load("models/regime_forecaster.joblib")
        except:
            logger.error("Models not found. Please run train_pipeline.py first.")
            raise FileNotFoundError

    def run_backtest(self, period="60d", interval="15m", transaction_cost=0.0001):
        """
        Runs a backtest over the specified period.
        """
        loader = DataLoader()
        df = loader.fetch_history(self.ticker, period=period, interval=interval)
        if df is None: return
        
        logger.info(f"Running backtest for {self.ticker} over {period}...")
        
        # 1. Feature Engineering
        df = FeatureEngineer.add_all_features(df)
        
        # 2. Inference
        regime_probs = self.ensemble.predict_proba(df)
        forecasts = self.forecaster.predict_weighted(df, regime_probs)
        
        # 3. Strategy Logic
        # Shift forecasts to avoid look-ahead (forecast at t is for t+4)
        # But we act at t based on forecast available at t.
        # Signal is 1 if forecast > threshold, -1 if < -threshold, else 0
        threshold = 0.0003
        df['signal'] = np.where(forecasts > threshold, 1, 0)
        df['signal'] = np.where(forecasts < -threshold, -1, df['signal'])
        
        # 4. Calculate Returns
        # We assume entry at close of bar t and hold for next bar(s). 
        # For simplicity: next bar return
        df['next_ret'] = df['log_ret'].shift(-1)
        
        # Strategy Return
        df['strat_ret'] = df['signal'] * df['next_ret']
        
        # Account for transaction costs on signal changes
        df['signal_change'] = df['signal'].diff().abs()
        df['strat_ret'] = df['strat_ret'] - (df['signal_change'] * transaction_cost)
        
        df.dropna(subset=['strat_ret'], inplace=True)
        
        # Cumulative Returns
        df['cum_market_ret'] = df['log_ret'].cumsum().apply(np.exp)
        df['cum_strat_ret'] = df['strat_ret'].cumsum().apply(np.exp)
        
        # 5. Metrics
        metrics = self._calculate_metrics(df)
        self._plot_results(df)
        
        # Save metrics to JSON
        with open("data/backtest_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Backtest metrics saved to data/backtest_metrics.json")
        
        return metrics

    def _calculate_metrics(self, df) -> Dict:
        total_ret = df['cum_strat_ret'].iloc[-1] - 1
        annualizing_factor = (252 * 6.5 * 4) # 15m bars in a year approx
        
        # Sharpe Ratio (annualized)
        daily_std = df['strat_ret'].std()
        sharpe = (df['strat_ret'].mean() / daily_std) * np.sqrt(annualizing_factor) if daily_std != 0 else 0
        
        # Drawdown
        cum_ret = df['cum_strat_ret']
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win Rate
        trades = df[df['signal'] != 0]
        win_rate = (trades['strat_ret'] > 0).mean() if len(trades) > 0 else 0
        
        metrics = {
            "Total Return": f"{total_ret:.2%}",
            "Annualized Sharpe": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Total Trades": len(trades)
        }
        
        logger.info("--- Backtest Metrics ---")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")
            
        return metrics

    def _plot_results(self, df):
        plt.figure(figsize=(12, 6))
        plt.plot(df['cum_market_ret'], label='Market (SPY)', alpha=0.7)
        plt.plot(df['cum_strat_ret'], label='Regime-Aware Strategy', linewidth=2)
        plt.title(f"Backtest Results: {self.ticker}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = "data/backtest_results.png"
        os.makedirs("data", exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Backtest plot saved to {output_path}")

if __name__ == "__main__":
    bt = Backtester()
    bt.run_backtest()
