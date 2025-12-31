import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XAULoader:
    """
    Loader for local XAUUSD CSV data.
    """
    def __init__(self, data_dir="src/xauusd_data"):
        # Resolve absolute path relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.data_dir = os.path.join(project_root, data_dir)
        
    def load_data(self, timeframe: str) -> pd.DataFrame:
        """
        Load XAUUSD data for a specific timeframe.
        Args:
            timeframe: '1m', '5m', '15m', '1h', '1d'
        """
        filename = f"XAU_{timeframe}_data.csv"
        path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return None
            
        logger.info(f"Loading {path}...")
        try:
            df = pd.read_csv(path, sep=';')
            
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Parse dates
            # Format: 2004.06.11 08:00
            df['datetime'] = pd.to_datetime(df['date'], format='%Y.%m.%d %H:%M')
            df.set_index('datetime', inplace=True)
            df.drop(columns=['date'], inplace=True)
            
            # Ensure unique index
            df = df[~df.index.duplicated(keep='first')]

            # Ensure UTC (assuming data is raw/local, but standardizing to UTC is good practice)
            # If data is already UTC, this just localizes it. 
            # Note: Forex data often starts at Sunday 5pm EST, but let's treat as UTC for pipeline consistency.
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            logger.info(f"Loaded {len(df)} rows for {timeframe}.")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

if __name__ == "__main__":
    loader = XAULoader()
    df_15m = loader.load_data("15m")
    if df_15m is not None:
        print(df_15m.head())
        print(df_15m.tail())
