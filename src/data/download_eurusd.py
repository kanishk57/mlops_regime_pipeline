import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import DataLoader

def download_eur_usd():
    loader = DataLoader()
    ticker = "EURUSD=X"
    
    # Intraday limits for YFinance:
    # 1h: max 730 days
    # 15m: max 60 days
    
    # 1. Fetch 1h
    df_1h = loader.fetch_history(ticker, period="730d", interval="1h")
    if df_1h is not None:
        save_path = "src/eurusd_data/EURUSD_1h_data.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_1h.to_csv(save_path)
        print(f"Saved 1h data to {save_path}")
    
    # 2. Fetch 15m
    df_15m = loader.fetch_history(ticker, period="60d", interval="15m")
    if df_15m is not None:
        save_path = "src/eurusd_data/EURUSD_15m_data.csv"
        df_15m.to_csv(save_path)
        print(f"Saved 15m data to {save_path}")

if __name__ == "__main__":
    download_eur_usd()
