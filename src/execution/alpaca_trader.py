import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
import logging
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.intraday_features import IntradayFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlpacaTrader:
    """
    Live/Paper Trading Bot using Alpaca API.
    Executes the Intraday Regime-Aware Strategy.
    """
    def __init__(self, symbol="XAUUSD", timeframe_regime="15Min", timeframe_forecast="5Min"):
        self.symbol = symbol
        self.tf_regime = timeframe_regime
        self.tf_forecast = timeframe_forecast
        
        # Load Credentials
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")
        self.base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca API credentials not found in environment variables.")
            raise ValueError("Missing Credentials")
            
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
        
        # Load Models
        logger.info("Loading models...")
        try:
            self.ensemble = joblib.load("models/regime_ensemble.joblib")
            self.forecaster = joblib.load("models/regime_forecaster.joblib")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
            
    def fetch_data(self, timeframe, limit=100) -> pd.DataFrame:
        """
        Fetch recent bars from Alpaca.
        """
        try:
            # Alpaca expects specific timeframe strings: '1Min', '5Min', '15Min'
            bars = self.api.get_bars(self.symbol, timeframe, limit=limit).df
            if bars.empty:
                return None
                
            # Rename columns to match our pipeline
            bars.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
            # Ensure index is datetime UTC
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize('UTC')
            return bars
        except Exception as e:
            logger.error(f"Error fetching data for {timeframe}: {e}")
            return None

    def trade_loop(self):
        logger.info(f"Starting Trading Loop for {self.symbol} (15m Pivot)...")
        
        while True:
            try:
                # 1. Fetch Data
                df_15m = self.fetch_data("15Min", limit=100)
                
                if df_15m is None:
                    logger.warning("Data fetch failed. Retrying in 1 min...")
                    time.sleep(60)
                    continue
                
                # 2. Features & Regime
                df_feat = IntradayFeatureEngineer.add_15m_features(df_15m)
                regime_probs = self.ensemble.predict_proba(df_feat)
                
                # Latest
                current_probs = regime_probs.iloc[-1]
                dominant_regime = current_probs.idxmax()
                
                # 3. Forecast
                # Attach regime for forecaster
                latest_row = df_feat.iloc[[-1]].copy()
                for c in current_probs.index:
                    latest_row[c] = current_probs[c]
                
                forecast = self.forecaster.predict_weighted(latest_row, latest_row[current_probs.index]).iloc[0]
                
                logger.info(f"Regime: {dominant_regime} | Forecast: {forecast:.6f}")
                
                # Check Session Close (e.g. 21:00 UTC)
                now = datetime.now()
                if now.hour >= 21:
                    logger.info("End of Session. Flattening.")
                    self.api.close_all_positions()
                    time.sleep(3600 * 8)
                    continue

                # 4. Execution (PIVOT: Only trade High Vol)
                if dominant_regime == 'p_high_vol':
                    self.execute(forecast)
                else:
                    logger.info(f"Skipping trade: {dominant_regime} is not p_high_vol")
                
                # Sleep until next 15m bar
                time.sleep(900)
                
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                time.sleep(60)

    def execute(self, forecast):
        try:
            position = 0
            try:
                pos = self.api.get_position(self.symbol)
                position = float(pos.qty)
            except:
                pass
            
            threshold = 0.0005
            
            if forecast > threshold and position <= 0:
                logger.info(f"Signal BUY. Current Pos: {position}")
                self.api.submit_order(symbol=self.symbol, qty=1, side='buy', type='market', time_in_force='day')
                
            elif forecast < -threshold and position >= 0:
                logger.info(f"Signal SELL. Current Pos: {position}")
                self.api.submit_order(symbol=self.symbol, qty=1, side='sell', type='market', time_in_force='day')
            else:
                logger.info("No Action (Threshold not met).")
                
        except Exception as e:
            logger.error(f"Execution Error: {e}")

if __name__ == "__main__":
    # Check env vars
    if not os.getenv("APCA_API_KEY_ID"):
        print("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in environment.")
        sys.exit(1)
        
    trader = AlpacaTrader(symbol="GLD") # Using GLD as proxy if XAUUSD not available
    trader.trade_loop()
