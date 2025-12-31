import os
import argparse
from loader import DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project root (2 levels up from src/data)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw")

def ingest_data(tickers, interval="15m", period="60d"):
    os.makedirs(DATA_DIR, exist_ok=True)
    loader = DataLoader()
    
    for ticker in tickers:
        logger.info(f"Ingesting {ticker}...")
        df = loader.fetch_history(ticker, period=period, interval=interval)
        
        if df is not None:
            # Clean ticker for filename (e.g., ^GSPC -> GSPC, EURUSD=X -> EURUSD)
            safe_ticker = ticker.replace("=X", "").replace("^", "").replace("-", "")
            filename = os.path.join(DATA_DIR, f"{safe_ticker}_{interval}.parquet")
            
            df.to_parquet(filename)
            logger.info(f"Saved {ticker} to {filename}")
        else:
            logger.warning(f"Failed to fetch {ticker}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM", "BTC-USD", "EURUSD=X"], help="Tickers to download")
    parser.add_argument("--interval", default="15m", help="Data interval (e.g., 1m, 5m, 15m)")
    parser.add_argument("--period", default="59d", help="Lookback period (yfinance limit for intraday is usually 60d)")
    
    args = parser.parse_args()
    
    ingest_data(args.tickers, args.interval, args.period)
