import yfinance as yf
import pandas as pd
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles fetching of financial market data.
    Supports Intraday (1m, 5m, 15m, 1h) and Daily resolutions.
    """
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    def fetch_history(self, ticker: str, period: str = "1mo", interval: str = "5m") -> Optional[pd.DataFrame]:
        """
        Fetches historical data from yfinance.
        
        Args:
            ticker: The symbol (e.g., 'SPY', 'EURUSD=X')
            period: The lookback period (e.g., '1d', '5d', '1mo', '1y')
            interval: The bar size (e.g., '1m', '5m', '15m', '1h')
            
        Returns:
            pd.DataFrame: OHLCV data with Datetime index, or None if failed.
        """
        logger.info(f"Fetching {interval} data for {ticker} over {period}...")
        
        try:
            # Use Ticker API for more reliable results
            tkr = yf.Ticker(ticker)
            df = tkr.history(period=period, interval=interval)
            
            if df is None or df.empty:
                logger.error(f"No data found for {ticker}.")
                return None
            
            # Ensure standard lowercase columns and single level
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            # Ensure Datetime index is timezone aware (UTC)
            if df.index.tz is None:
                 df.index = df.index.tz_localize('UTC')
            else:
                 df.index = df.index.tz_convert('UTC')

            logger.info(f"Successfully fetched {len(df)} rows for {ticker}.")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None

if __name__ == "__main__":
    # Smoke test
    loader = DataLoader()
    data = loader.fetch_history(ticker="SPY", period="5d", interval="15m")
    if data is not None:
        print(data.head())
        print(data.tail())
