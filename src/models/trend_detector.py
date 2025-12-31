import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class TrendDetector:
    """
    Regime Classifier based on Trend Strength.
    Uses linear regression slope + directional strength to assign trend/range probabilities.
    """
    
    def __init__(self, window=20):
        """
        Args:
            window: Lookback for trend calculation (e.g., 20 bars = 5 hours on 15m)
        """
        self.window = window
        self.slope_threshold = 0.001  # Minimum absolute slope to call it trending
        
    def fit(self, df: pd.DataFrame):
        """
        Learn trend distribution from training data.
        """
        slopes = self._calculate_slopes(df)
        
        # Store slope distribution stats
        self.slope_std = slopes.abs().std()
        self.slope_mean = slopes.abs().mean()
        
        return self
        
    def _calculate_slopes(self, df: pd.DataFrame) -> pd.Series:
        """Calculate rolling linear regression slopes"""
        slopes = pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        
        for i in range(self.window, len(df)):
            y = prices[i-self.window:i]
            X = np.arange(self.window).reshape(-1, 1)
            
            lr = LinearRegression()
            lr.fit(X, y)
            slopes.iloc[i] = lr.coef_[0]
        
        return slopes
    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities based on trend strength.
        
        Returns:
            DataFrame with columns: p_uptrend, p_downtrend, p_ranging
        """
        slopes = self._calculate_slopes(df)
        
        probs = pd.DataFrame(index=df.index)
        
        # Normalize slope by historical std
        norm_slope = slopes / (self.slope_std if self.slope_std > 0 else 1.0)
        
        # Uptrend probability (positive slope)
        probs['p_uptrend'] = np.clip(norm_slope, 0, 3) / 3.0
        
        # Downtrend probability (negative slope)
        probs['p_downtrend'] = np.clip(-norm_slope, 0, 3) / 3.0
        
        # Ranging probability (weak slope)
        probs['p_ranging'] = 1.0 - probs['p_uptrend'] - probs['p_downtrend']
        probs['p_ranging'] = probs['p_ranging'].clip(lower=0)
        
        # Temporal smoothing
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
        
        detector = TrendDetector()
        detector.fit(df)
        probs = detector.predict_proba(df)
        
        print("Trend Regime Probabilities (last 10 rows):")
        print(probs.tail(10))
