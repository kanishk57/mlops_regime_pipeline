from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
import sys
from pydantic import BaseModel
from typing import List

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Regime-Aware Forecaster API")

# Model state
MODEL_BUNDLE = {}

class Bar(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictRequest(BaseModel):
    ticker: str
    bars: List[Bar]

@app.on_event("startup")
def load_models():
    bundle_path = "models/regime_forecaster.joblib"
    ensemble_path = "models/regime_ensemble.joblib"
    
    if not os.path.exists(bundle_path) or not os.path.exists(ensemble_path):
        logger.error("Model files not found. Run train_pipeline first.")
        return
        
    MODEL_BUNDLE['forecaster'] = joblib.load(bundle_path)
    MODEL_BUNDLE['ensemble'] = joblib.load(ensemble_path)
    logger.info("Models loaded successfully.")

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "forecaster" in MODEL_BUNDLE}

@app.post("/predict")
def predict(request: PredictRequest):
    if "forecaster" not in MODEL_BUNDLE:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    # 1. Convert to DataFrame
    df = pd.DataFrame([b.dict() for b in request.bars])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # 2. Add Features
    # Note: For production, we'd need enough historical bars (96+) to calc features
    if len(df) < 96:
         raise HTTPException(status_code=400, detail="Need at least 96 bars for full feature engineering")
         
    df_features = FeatureEngineer.add_all_features(df)
    
    # 3. Regime Inference
    ensemble = MODEL_BUNDLE['ensemble']
    regime_probs = ensemble.predict_proba(df_features)
    
    # 4. Forecast Inference
    forecaster = MODEL_BUNDLE['forecaster']
    forecasts = forecaster.predict_weighted(df_features, regime_probs)
    
    # 5. Return latest forecast
    latest_time = forecasts.index[-1]
    latest_val = forecasts.iloc[-1]
    latest_regime = regime_probs.idxmax(axis=1).iloc[-1]
    
    return {
        "ticker": request.ticker,
        "timestamp": str(latest_time),
        "forecast_1h_ret": float(latest_val),
        "dominant_regime": latest_regime,
        "regime_probabilities": regime_probs.iloc[-1].to_dict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
