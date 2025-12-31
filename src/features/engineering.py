import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Central repository for feature calculations.
    Used by both Offline ETL and Online Inference interactions to ensure strict symmetry.
    """
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, price_col='close') -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1))
        return df

    @staticmethod
    def calculate_volatility(df: pd.DataFrame, window=20) -> pd.DataFrame:
        df = df.copy()
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
        df[f'vol_{window}'] = df['log_ret'].rolling(window=window).std()
        return df

    @staticmethod
    def calculate_trend_strength(df: pd.DataFrame, window=14) -> pd.DataFrame:
        df = df.copy()
        df[f'roc_{window}'] = df['close'].pct_change(window).abs()
        return df

    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = FeatureEngineer.calculate_returns(df)
        df = FeatureEngineer.calculate_volatility(df, window=12)
        df = FeatureEngineer.calculate_volatility(df, window=96)
        df = FeatureEngineer.calculate_trend_strength(df, window=20)
        return df.dropna()
