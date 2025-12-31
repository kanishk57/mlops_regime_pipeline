import pandas as pd
import numpy as np
from typing import Tuple

class VolatilityClassifier:
    """
    Regime Classifier based on Rolling Realized Volatility.
    Assigns regime probabilities based on current volatility percentile.
    """
    
    def __init__(self, short_window=12, long_window=96):
        """
        Args:
            short_window: Lookback for volatility calculation (e.g., 12 bars = 3 hours on 15m)
            long_window: Lookback for percentile calculation (e.g., 96 bars = 1 day on 15m)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.thresholds = {'low': 0.33, 'high': 0.67}  # Percentile thresholds
        
    def fit(self, df: pd.DataFrame):
        """
        Learn volatility distribution from training data.
        """
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        vol = df['log_ret'].rolling(window=self.short_window).std()
        
        # Store percentile thresholds
        self.low_threshold = vol.quantile(self.thresholds['low'])
        self.high_threshold = vol.quantile(self.thresholds['high'])
        
        return self
        
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities based on current volatility.
        
        Returns:
            DataFrame with columns: p_low_vol, p_medium_vol, p_high_vol
        """
        if 'log_ret' not in df.columns:
            df = df.copy()
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        vol = df['log_ret'].rolling(window=self.short_window).std()
        
        # Soft assignment using z-score relative to thresholds
        probs = pd.DataFrame(index=df.index)
        
        # Simple approach: discrete assignment with small smoothing
        probs['p_low_vol'] = (vol < self.low_threshold).astype(float)
        probs['p_high_vol'] = (vol > self.high_threshold).astype(float)
        probs['p_medium_vol'] = 1.0 - probs['p_low_vol'] - probs['p_high_vol']
        
        # Temporal smoothing (exponential moving average)
        for col in probs.columns:
            probs[col] = probs[col].ewm(span=5).mean()
        
        # Renormalize
        probs = probs.div(probs.sum(axis=1), axis=0)
        
        return probs

if __name__ == "__main__":
    # Smoke test
    import sys
    sys.path.append('..')
    from data.loader import DataLoader
    from features.engineering import FeatureEngineer
    
    loader = DataLoader()
    df = loader.fetch_history("SPY", period="30d", interval="15m")
    
    if df is not None:
        df = FeatureEngineer.add_all_features(df)
        
        clf = VolatilityClassifier()
        clf.fit(df)
        probs = clf.predict_proba(df)
        
        print("Volatility Regime Probabilities (last 10 rows):")
        print(probs.tail(10))
