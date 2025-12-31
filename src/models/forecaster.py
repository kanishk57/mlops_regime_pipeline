import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import logging

logger = logging.getLogger(__name__)

class RegimeForecaster:
    """
    Forecasting system that uses regime-specific expert models.
    """
    
    def __init__(self, regimes=['p_high_vol', 'p_trending', 'p_ranging'], feature_cols=None):
        self.regimes = regimes
        self.models = {}
        if feature_cols:
            self.feature_cols = feature_cols
        else:
            self.feature_cols = [
                'log_ret', 'vol_12', 'vol_96', 'roc_20'
            ]
        
    def train(self, ticker="SPY_15m"):
        """
        Train a model for each regime using the labeled data.
        """
        for regime in self.regimes:
            path = f"data/processed/{ticker}_{regime}.parquet"
            if not os.path.exists(path):
                logger.warning(f"No data for regime {regime} at {path}")
                continue
                
            df = pd.read_parquet(path)
            if len(df) < 50:
                logger.warning(f"Too few samples for {regime} ({len(df)})")
                continue
                
            X = df[self.feature_cols]
            y = df['target']
            
            logger.info(f"Training expert for {regime} with {len(df)} samples...")
            
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            model.fit(X, y)
            self.models[regime] = model
            
            # Save model
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, f"models/forecaster_{regime}.joblib")
            
        return self

    def predict_weighted(self, df: pd.DataFrame, regime_probs: pd.DataFrame) -> pd.Series:
        """
        Combine experts using regime probabilities.
        """
        X = df[self.feature_cols]
        final_forecast = pd.Series(0.0, index=df.index)
        
        for regime, model in self.models.items():
            if regime in regime_probs.columns:
                pred = model.predict(X)
                final_forecast += pred * regime_probs[regime]
                
        return final_forecast

if __name__ == "__main__":
    # Smoke test
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    forecaster = RegimeForecaster()
    forecaster.train()
    
    # Reload and test prediction
    df = pd.read_parquet("data/processed/SPY_15m_labeled.parquet")
    regime_probs = df[['p_high_vol', 'p_trending', 'p_ranging']]
    
    forecasts = forecaster.predict_weighted(df, regime_probs)
    print("\nFinal Weighted Forecasts (last 10):")
    print(forecasts.tail(10))
    print("\nActual Targets (last 10):")
    print(df['target'].tail(10))
