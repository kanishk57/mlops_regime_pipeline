import pandas as pd
import numpy as np
try:
    from .vol_classifier import VolatilityClassifier
    from .trend_detector import TrendDetector
    from .hmm_detector import HMMRegimeDetector
except ImportError:
    from vol_classifier import VolatilityClassifier
    from trend_detector import TrendDetector
    from hmm_detector import HMMRegimeDetector

class RegimeEnsemble:
    """
    Ensemble combining multiple regime detection models.
    Produces final regime probabilities via weighted averaging.
    """
    
    def __init__(self, weights=None):
        """
        Args:
            weights: Dict with keys 'vol', 'trend', 'hmm'. If None, uses default weights.
        """
        self.vol_classifier = VolatilityClassifier()
        self.trend_detector = TrendDetector()
        self.hmm_detector = HMMRegimeDetector(n_components=3)
        
        if weights is None:
            self.weights = {'vol': 0.3, 'trend': 0.3, 'hmm': 0.4}
        else:
            self.weights = weights
            
    def fit(self, df: pd.DataFrame):
        """
        Fit all ensemble components.
        """
        self.vol_classifier.fit(df)
        self.trend_detector.fit(df)
        self.hmm_detector.fit(df)
        return self
        
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict final regime probabilities.
        """
        try:
            vol_probs = self.vol_classifier.predict_proba(df)
            trend_probs = self.trend_detector.predict_proba(df)
            hmm_probs = self.hmm_detector.predict_proba(df)
            
            final_probs = pd.DataFrame(index=df.index)
            
            # Use .values.flatten() to be absolutely sure
            def get_vals(df_in, col, target_index):
                series = df_in[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return series.reindex(target_index).fillna(0).values.flatten()

            v_high = get_vals(vol_probs, 'p_high_vol', df.index)
            v_med = get_vals(vol_probs, 'p_medium_vol', df.index)
            v_low = get_vals(vol_probs, 'p_low_vol', df.index)
            
            t_range = get_vals(trend_probs, 'p_ranging', df.index)
            t_up = get_vals(trend_probs, 'p_uptrend', df.index)
            t_down = get_vals(trend_probs, 'p_downtrend', df.index)
            
            h_state0 = get_vals(hmm_probs, 'p_hmm_state_0', df.index)
            h_state1 = get_vals(hmm_probs, 'p_hmm_state_1', df.index)
            h_state2 = get_vals(hmm_probs, 'p_hmm_state_2', df.index)

            final_probs['p_high_vol'] = (
                v_high * self.weights['vol'] + 
                (1 - t_range) * self.weights['trend'] * 0.5 +
                h_state2 * self.weights['hmm']
            )
            
            final_probs['p_trending'] = (
                (t_up + t_down) * self.weights['trend'] +
                v_med * self.weights['vol'] * 0.5 +
                h_state1 * self.weights['hmm']
            )
            
            final_probs['p_ranging'] = (
                t_range * self.weights['trend'] +
                v_low * self.weights['vol'] +
                h_state0 * self.weights['hmm']
            )
            
            # Renormalize
            final_probs = final_probs.div(final_probs.sum(axis=1), axis=0).fillna(0)
            
            return final_probs
        except Exception as e:
            import traceback
            print(f"ERROR in RegimeEnsemble.predict_proba: {e}")
            traceback.print_exc()
            raise e
    
    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict most likely regime (hard classification).
        """
        probs = self.predict_proba(df)
        return probs.idxmax(axis=1)

if __name__ == "__main__":
    # Smoke test
    import sys
    sys.path.append('..')
    from data.loader import DataLoader
    from features.engineering import FeatureEngineer
    
    loader = DataLoader()
    df = loader.fetch_history("SPY", period="30d", interval="15m")
    
    if df is not None:
        df = FeatureEngineer.add_all_features(df)
        
        ensemble = RegimeEnsemble()
        ensemble.fit(df)
        probs = ensemble.predict_proba(df)
        regimes = ensemble.predict_regime(df)
        
        print("Final Regime Probabilities (last 10 rows):")
        print(probs.tail(10))
        print("\nMost Likely Regimes (last 10):")
        print(regimes.tail(10))
