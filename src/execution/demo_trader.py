import pandas as pd
import numpy as np
import time
import sys
import os
import joblib

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import DataLoader
from features.engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoTrader:
    """
    Simulated Paper Trading Engine.
    Polls data, gets forecasts, and 'executes' trades.
    """
    def __init__(self, ticker="SPY", initial_capital=100000):
        self.ticker = ticker
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        
        # Load local models for simplicity in demo
        self.ensemble = joblib.load("models/regime_ensemble.joblib")
        self.forecaster = joblib.load("models/regime_forecaster.joblib")
        
    def run_simulation(self, steps=10):
        """
        Simulate real-time steps using recent 15m data.
        """
        loader = DataLoader()
        # Fetch data (longer period to have enough for features)
        df_all = loader.fetch_history(self.ticker, period="30d", interval="15m")
        if df_all is None: return
        
        logger.info(f"Starting Demo Trading for {self.ticker}...")
        
        # We simulate the last 'steps' bars
        for i in range(len(df_all) - steps, len(df_all)):
            # "Context" is all data up to i
            df_context = df_all.iloc[:i+1]
            current_price = df_context['close'].iloc[-1]
            current_time = df_context.index[-1]
            
            # 1. Feature Engineering
            df_feat = FeatureEngineer.add_all_features(df_context)
            if len(df_feat) == 0: continue
            
            # 2. Inference
            regime_probs = self.ensemble.predict_proba(df_feat)
            forecast = self.forecaster.predict_weighted(df_feat, regime_probs).iloc[-1]
            dominant_regime = regime_probs.idxmax(axis=1).iloc[-1]
            
            # 3. Policy (Threshold: 0.03% expected 1h return)
            threshold = 0.0003
            signal = 0
            if forecast > threshold: signal = 1
            elif forecast < -threshold: signal = -1
            
            # 4. Execution
            self._execute(signal, current_price, current_time, dominant_regime, forecast)
            
        self.report()

    def _execute(self, signal, price, timestamp, regime, forecast):
        if signal == 1 and self.position <= 0:
            logger.info(f"[{timestamp}] BUY @ {price:.2f} | Forecast: {forecast:.5f} | Regime: {regime}")
            self.position = 1
            self.trades.append({'time': timestamp, 'type': 'BUY', 'price': price, 'regime': regime})
        elif signal == -1 and self.position >= 0:
            logger.info(f"[{timestamp}] SELL @ {price:.2f} | Forecast: {forecast:.5f} | Regime: {regime}")
            self.position = -1
            self.trades.append({'time': timestamp, 'type': 'SELL', 'price': price, 'regime': regime})

    def report(self):
        logger.info("--- Simulation Report ---")
        logger.info(f"Initial Capital: $100,000")
        logger.info(f"Total Trades: {len(self.trades)}")
        # Simple PnL calculation omitted for brevity in shell
        
if __name__ == "__main__":
    trader = DemoTrader()
    trader.run_simulation(steps=20)
