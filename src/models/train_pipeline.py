import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import pandas as pd
import os
import joblib
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import DataLoader
from features.engineering import FeatureEngineer
from models.regime_ensemble import RegimeEnsemble
from models.forecaster import RegimeForecaster
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_pipeline(ticker="SPY"):
    mlflow.set_experiment("RegimeAwareForecasting")
    
    with mlflow.start_run():
        # 1. Load Data
        loader = DataLoader()
        df = loader.fetch_history(ticker, period="60d", interval="15m")
        if df is None: return
        
        # 2. Features
        df = FeatureEngineer.add_all_features(df)
        
        # 3. Regime Ensemble
        logger.info("Fitting Regime Ensemble...")
        ensemble = RegimeEnsemble()
        ensemble.fit(df)
        
        # 4. Label & Split
        logger.info("Predicting regimes...")
        regime_probs = ensemble.predict_proba(df)
        
        # Merge regime probabilities using index join
        df_labeled = df.join(regime_probs, how='inner')
        logger.info(f"Joined data shape: {df_labeled.shape}")
        
        df_labeled['dominant_regime'] = df_labeled[regime_probs.columns].idxmax(axis=1)
        
        # 5. Create Target
        # Shift target by -4 to predict 1 hour ahead
        df_labeled['target'] = df_labeled['log_ret'].shift(-4).rolling(window=4).sum()
        df_labeled = df_labeled.dropna()
        
        # 6. Train Forecaster
        logger.info("Training Forecasters...")
        forecaster = RegimeForecaster()
        os.makedirs("data/processed", exist_ok=True)
        for r in ['p_high_vol', 'p_trending', 'p_ranging']:
            rdf = df_labeled[df_labeled['dominant_regime'] == r]
            rdf.to_parquet(f"data/processed/{ticker}_15m_{r}.parquet")
            
        forecaster.train(ticker=f"{ticker}_15m")
        
        # 6. Log Artifacts to MLflow
        mlflow.log_params({
            "ticker": ticker,
            "interval": "15m",
            "regime_weights": ensemble.weights
        })
        
        # Save ensemble and forecaster
        os.makedirs("models", exist_ok=True)
        joblib.dump(ensemble, "models/regime_ensemble.joblib")
        joblib.dump(forecaster, "models/regime_forecaster.joblib")
        
        mlflow.log_artifacts("models", artifact_path="model_bundle")
        
        # 7. Mock Evaluation
        preds = forecaster.predict_weighted(df_labeled, df_labeled[regime_probs.columns])
        mse = ((preds - df_labeled['target'])**2).mean()
        mlflow.log_metric("mse", mse)
        
        logger.info(f"Pipeline complete. MSE: {mse}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    run_full_pipeline()
