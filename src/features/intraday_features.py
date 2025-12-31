import pandas as pd
import numpy as np

class IntradayFeatureEngineer:
    """
    Feature Engineering specifically tailored for Intraday XAUUSD.
    Enforces strict lookback constraints to avoid lag.
    """
    
    @staticmethod
    def add_15m_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Features for Regime + Forecasting (15m Primary).
        Constraint: Max lookback = 48 bars (~12 hours).
        """
        df = df.copy()
        
        # Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum / Trend
        df['roc_4'] = df['close'].pct_change(4)   # 1 hour
        df['roc_12'] = df['close'].pct_change(12) # 3 hours
        df['roc_48'] = df['close'].pct_change(48) # 12 hours (session trend)
        
        # Volatility
        df['vol_12'] = df['log_ret'].rolling(window=12).std()
        df['vol_48'] = df['log_ret'].rolling(window=48).std()
        df['relative_vol'] = df['vol_12'] / df['vol_48'].replace(0, np.nan)
        
        # Simple RSI-proxy (Momentum / Volatility)
        # Real RSI is costlier, this is a vector-friendly approx
        up = df['log_ret'].clip(lower=0)
        down = -df['log_ret'].clip(upper=0)
        ma_up = up.rolling(14).mean()
        ma_down = down.rolling(14).mean()
        df['rsi_proxy'] = ma_up / (ma_down + 1e-9)
        
        # Range / ATR proxy
        hl = (df['high'] - df['low']) / df['close']
        df['atr_12'] = hl.rolling(12).mean()
        
        return df.dropna()

    @staticmethod
    def add_5m_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Features for Forecasting (5m).
        Constraint: Max lookback = 60 bars (5 hours).
        """
        df = df.copy()
        
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum signals
        df['mom_12'] = df['close'].pct_change(12) # 1 hour
        df['mom_36'] = df['close'].pct_change(36) # 3 hours
        
        # Volatility
        df['vol_24'] = df['log_ret'].rolling(24).std() # 2 hours
        
        # Relative Volume
        df['vol_ratio'] = df['volume'] / df['volume'].rolling(60).mean()
        
        return df.dropna()
