import pandas as pd
import numpy as np
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)

class HMMRegimeDetector:
    """
    Hidden Markov Model (HMM) for regime detection.
    Unsupervised learning of market states based on returns and volatility.
    """
    
    def __init__(self, n_components=3, covariance_type="full"):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type="diag", 
            n_iter=100,
            random_state=42
        )
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """
        Fit HMM using log returns and rolling volatility.
        """
        data = df.copy()
        if 'log_ret' not in data.columns:
            data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
            
        if 'vol_12' not in data.columns:
            data['vol_12'] = data['log_ret'].rolling(window=12).std()
        
        # Avoid dropping NaNs to maintain length, use fillna(0) for training too
        # or better, just slice the valid part for fit but keep track of length.
        X_df = data[['log_ret', 'vol_12']].fillna(0)
        X = X_df.values
        
        logger.info(f"Fitting HMM with {len(X)} samples...")
        self.model.fit(X)
        self.is_fitted = True
        return self
        
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict state probabilities.
        """
        if not self.is_fitted:
            raise ValueError("HMM Model not fitted.")
            
        data = df.copy()
        if 'log_ret' not in data.columns:
            data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
        
        if 'vol_12' not in data.columns:
            data['vol_12'] = data['log_ret'].rolling(window=12).std()
        
        # Fill NaNs for prediction (HMM doesn't like them)
        X_df = data[['log_ret', 'vol_12']].fillna(0)
        X = X_df.values
        
        # HMM outputs probabilities for each state
        logger.info(f"HMM predict_proba input X shape: {X.shape}")
        probs = self.model.predict_proba(X)
        logger.info(f"HMM predict_proba output probs shape: {probs.shape}")
        
        prob_df = pd.DataFrame(
            probs, 
            index=df.index, 
            columns=[f'p_hmm_state_{i}' for i in range(self.n_components)]
        )
        
        # Temporal smoothing
        for col in prob_df.columns:
            prob_df[col] = prob_df[col].ewm(span=3).mean()
            
        # Renormalize
        prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
        
        return prob_df

if __name__ == "__main__":
    # Smoke test
    import sys
    import os
    # Add parent directory to path to import data and features
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.loader import DataLoader
    from features.engineering import FeatureEngineer
    
    loader = DataLoader()
    df = loader.fetch_history("SPY", period="30d", interval="15m")
    
    if df is not None:
        df = FeatureEngineer.add_all_features(df)
        
        hmm_detector = HMMRegimeDetector(n_components=3)
        hmm_detector.fit(df)
        probs = hmm_detector.predict_proba(df)
        
        print("HMM Regime Probabilities (last 10 rows):")
        print(probs.tail(10))
