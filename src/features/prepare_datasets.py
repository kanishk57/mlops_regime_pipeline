import os
import pandas as pd
import numpy as np
import sys
# Add src and src/models to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from data.loader import DataLoader
from features.engineering import FeatureEngineer
from models.regime_ensemble import RegimeEnsemble
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_regime_datasets(ticker="SPY_15m"):
    raw_path = f"data/raw/{ticker}.parquet"
    if not os.path.exists(raw_path):
        logger.error(f"Raw data not found at {raw_path}")
        return

    df = pd.read_parquet(raw_path)
    logger.info(f"Loaded {len(df)} rows of raw data.")

    # 1. Feature Engineering
    df = FeatureEngineer.add_all_features(df)
    
    # 2. Fit Regime Ensemble
    # In a production system, we'd load a pre-trained ensemble.
    # Here we fit it on the whole dataset for labeling.
    ensemble = RegimeEnsemble()
    ensemble.fit(df)
    
    # 3. Label Data
    regime_probs = ensemble.predict_proba(df)
    df = pd.concat([df, regime_probs], axis=1)
    df['dominant_regime'] = regime_probs.idxmax(axis=1)
    
    # 4. Create Target (Next 4 bars log return - 1 hour forecast)
    df['target'] = df['log_ret'].shift(-4).rolling(window=4).sum()
    df = df.dropna()

    # 5. Save Processed Data
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(f"data/processed/{ticker}_labeled.parquet")
    logger.info(f"Saved labeled data to data/processed/{ticker}_labeled.parquet")

    # 6. Split and Save per Regime
    for regime in ['p_high_vol', 'p_trending', 'p_ranging']:
        regime_df = df[df['dominant_regime'] == regime]
        regime_df.to_parquet(f"data/processed/{ticker}_{regime}.parquet")
        logger.info(f"Saved {len(regime_df)} rows for regime {regime}")

if __name__ == "__main__":
    prepare_regime_datasets()
