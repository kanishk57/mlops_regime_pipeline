import pandas as pd
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ForexLoader:
    """
    Loader for local EURUSD CSV files with varying formats.
    """
    def __init__(self, data_dir: str = "src/eurusd_data"):
        self.data_dir = data_dir

    def load_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Loads local EURUSD data for the specified timeframe.
        """
        files = {
            "15m": "EURUSD M15 2010-2016.csv",
            "1h": "EURUSD_1H_2020-2024.csv",
            "5m": "EURUSD_5M_2020-2024.csv"
        }
        
        if timeframe not in files:
            logger.error(f"Timeframe {timeframe} not supported.")
            return None
            
        file_path = os.path.join(self.data_dir, files[timeframe])
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        logger.info(f"Loading {timeframe} data from {file_path}...")
        
        try:
            # Detect separator (usually comma or semicolon)
            # Most of these look like comma
            df = pd.read_csv(file_path, sep=',')
            
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Find datetime column
            date_cols = ['time', 'datetime', 'time (utc)']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                # Fallback to first column if it looks like a date
                date_col = df.columns[0]
                
            df['datetime'] = pd.to_datetime(df[date_col])
            df.set_index('datetime', inplace=True)
            
            # Keep only OHLCV
            keep_cols = ['open', 'high', 'low', 'close', 'volume', 'tick_volume']
            df = df[[c for c in df.columns if c in keep_cols]]
            
            # Standardize volume name
            if 'tick_volume' in df.columns and 'volume' not in df.columns:
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            # Sort and deduplicate
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            # Localize to UTC if not aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
            logger.info(f"Successfully loaded {len(df)} rows for {timeframe}.")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {timeframe} data: {e}")
            return None
