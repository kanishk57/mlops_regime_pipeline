import pandas as pd
import os
import sys
import joblib
import logging
import numpy as np
import mlflow
import shutil

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.forex_loader import ForexLoader
from features.intraday_features import IntradayFeatureEngineer
from models.regime_ensemble import RegimeEnsemble
from models.forecaster import RegimeForecaster

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_walk_forward_eurusd():
    mlflow.set_experiment("EURUSD_Regime_Pivot")
    
    loader = ForexLoader()
    
    # 1. Load Data
    # We'll use 1h as it is the most recent (2020-2024)
    logger.info("Loading EURUSD 1h Data...")
    df_1h = loader.load_data("1h")
    
    if df_1h is None:
        logger.error("Failed to load EURUSD data.")
        return

    # 2. Features
    logger.info("Generating Features...")
    df_feat = IntradayFeatureEngineer.add_15m_features(df_1h)
    
    # 3. Create Target
    # Target: Return over next 4 hours
    df_feat['target'] = df_feat['log_ret'].shift(-4).rolling(4).sum()
    df_feat.dropna(inplace=True)
    
    # 4. Walk-Forward Setup
    # Train: 52 weeks (1 year), Test: 4 weeks (1 month)
    BARS_PER_WEEK = 24 * 5 
    TRAIN_SIZE = 52 * BARS_PER_WEEK
    TEST_SIZE = 4 * BARS_PER_WEEK
    STRIDE = TEST_SIZE
    
    # Run on the most recent 2 years
    start_idx = max(0, len(df_feat) - (24 * STRIDE + TRAIN_SIZE))
    
    feature_cols = [
        'log_ret', 'roc_4', 'roc_12', 'roc_48', 
        'vol_12', 'relative_vol', 'rsi_proxy', 'atr_12'
    ]
    
    results = []
    step = 0
    
    logger.info(f"Starting EURUSD 1H Walk-Forward... Start Index: {start_idx}")
    
    try:
        while start_idx + TRAIN_SIZE + TEST_SIZE < len(df_feat):
            step += 1
            
            # Slicing
            train_df = df_feat.iloc[start_idx : start_idx + TRAIN_SIZE].copy()
            test_df = df_feat.iloc[start_idx + TRAIN_SIZE : start_idx + TRAIN_SIZE + TEST_SIZE].copy()
            
            # Regime Ensemble
            ensemble = RegimeEnsemble()
            ensemble.fit(train_df)
            
            train_regimes = ensemble.predict_proba(train_df)
            test_regimes = ensemble.predict_proba(test_df)
            
            # Forecaster
            train_labeled = train_df.join(train_regimes)
            test_labeled = test_df.join(test_regimes)
            train_labeled['dominant_regime'] = train_labeled[train_regimes.columns].idxmax(axis=1)
            
            forecaster = RegimeForecaster(feature_cols=feature_cols)
            
            temp_dir = f"data/processed/temp_eur_{step}"
            os.makedirs(temp_dir, exist_ok=True)
            for r in ['p_high_vol', 'p_trending', 'p_ranging']:
                subset = train_labeled[train_labeled['dominant_regime'] == r]
                if not subset.empty:
                    subset.to_parquet(f"{temp_dir}/train_{r}.parquet")
            
            forecaster.train(ticker=f"temp_eur_{step}/train")
            
            # Predict
            preds = forecaster.predict_weighted(test_labeled, test_labeled[train_regimes.columns])
            
            # Save
            step_res = pd.DataFrame({
                'datetime': test_labeled.index,
                'close': test_labeled['close'].values,
                'target': test_labeled['target'].values,
                'forecast': preds.values,
                'regime': test_labeled[train_regimes.columns].idxmax(axis=1).values
            })
            results.append(step_res)
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Step {step}: {test_df.index[0].date()} -> Net LogRet Summary...")
            
            start_idx += STRIDE
            
    except Exception as e:
        logger.error(f"Error in WF: {e}")
        import traceback
        logger.error(traceback.format_exc())

    if not results:
        logger.error("No results generated.")
        return

    full_df = pd.concat(results).sort_values('datetime').set_index('datetime')
    full_df['next_ret'] = np.log(full_df['close'].shift(-1) / full_df['close'])
    full_df.dropna(inplace=True)
    
    # --- STRATEGY: SNIPER (High Vol Mom) + OPTIONAL REVERSION ---
    # We saw in XAUUSD that High Vol Mom was the only one with positive gross alpha.
    # Let's see how EURUSD behaves.
    
    threshold = 0.0005 # 5 bps
    full_df['signal_mom'] = 0
    full_df.loc[(full_df['regime'] == 'p_high_vol') & (full_df['forecast'] > threshold), 'signal_mom'] = 1
    full_df.loc[(full_df['regime'] == 'p_high_vol') & (full_df['forecast'] < -threshold), 'signal_mom'] = -1
    
    # Calculate Costs (EURUSD spread is extremely low, e.g. 0.1 - 0.5 pips)
    # 0.5 pips at 1.10 is ~0.00005. Let's use 0.5 bps (0.00005) for EURUSD.
    t_cost = 0.00005 
    
    def calc_metrics(df, signal_col, title):
        df = df.copy()
        df['trades'] = df[signal_col].diff().abs()
        df['strat_ret'] = df[signal_col] * df['next_ret'] - (df['trades'] * t_cost)
        df['cum_ret'] = df['strat_ret'].cumsum().apply(np.exp)
        
        total_ret = df['cum_ret'].iloc[-1] - 1
        sharpe = (df['strat_ret'].mean() / df['strat_ret'].std()) * np.sqrt(252 * 24) if df['strat_ret'].std() > 0 else 0
        
        rm = df['cum_ret'].cummax()
        dd = (df['cum_ret'] - rm) / rm
        max_dd = dd.min()
        
        print(f"\n--- {title} ---")
        print(f"Total Return: {total_ret:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Trade Count: {int(df['trades'].sum()/2)}")
        
        # Performance by Regime
        df['gross_ret'] = df[signal_col] * df['next_ret']
        regime_perf = df.groupby('regime')['gross_ret'].sum()
        print("Gross Alpha by Regime:")
        print(regime_perf)

    calc_metrics(full_df, 'signal_mom', "EURUSD 1H - SNIPER (HIGH VOL MOM)")
    
    full_df.to_csv("data/eur_usd_1h_backtest.csv")

if __name__ == "__main__":
    train_walk_forward_eurusd()
