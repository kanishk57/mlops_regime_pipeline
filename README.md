# Regime-Aware Intraday Trading Pipeline (XAUUSD & EURUSD)

## 📋 Project Overview
This project implements an end-to-end quantitative research and trading framework designed to exploit market regime transitions in high-liquidity assets (Gold and Forex). The system employs a **Regime-Conditioned Forecasting** architecture, where specialized expert models are trained and ensembled based on the current market environment.

### Key Components:
- **Data Layer**: Custom loaders for local proprietary CSV data and Yahoo Finance integration.
- **Feature Engineering**: Strict lookback-constrained features (Momentum, Volatility, RSI-proxy, ATR) to prevent data leakage.
- **Regime Ensemble**: A tri-factor model combining:
  - **Volatility Classifier**: Percentile-based regime detection.
  - **Trend Detector**: Linear regression slope analysis.
  - **HMM (Hidden Markov Model)**: Stochastic state identification.
- **Forecasting Layer**: Regime-weighted LightGBM ensemble predicting next-bar log returns.
- **Verification**: Walk-forward validation (Weekly and Monthly increments) with realistic transaction cost modeling.

---

## 📊 Research Outcomes & Backtest Results

| Experiment | Timeframe | Gross Alpha (Best Regime) | Net PnL (After Costs) | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **XAUUSD v1** | 5m / 15m | +1.20% | **Negative** | Microstructure noise and high trade frequency (35+/day) created unrecoverable drag. |
| **XAUUSD v2 (Sniper)** | 15m | +2.46% | **Breakeven** | Trading only in High Volatility regimes showed positive gross alpha, but thin net edge. |
| **EURUSD 1H** | 1H | +0.00% | **Negative** | Even on slower timeframes, standard OHLC features struggle to overcome market efficiency in directional forecasting. |

### 🛠️ Key Technical Achievements:
- **Zero-Leakage Pipeline**: Verified walk-forward logic ensures no look-ahead bias.
- **Modular Infrastructure**: Support for any asset with minimal configuration changes.
- **MLOps Integration**: Full experiment tracking via MLflow.
- **Alpaca Trading Ready**: Implemented `AlpacaTrader` for paper/live execution.

---

## 🏁 Final Conclusion
The research phase has successfully concluded with a **Negative Trading Result**. 

**Primary Finding**: For intraday Gold and EURUSD, directional forecasting based solely on price-derived features (OHLC) does not provide a sufficient statistical edge to overcome modern transaction costs and institutional competition.

**Success Metric**: The framework correctly identified when "not to trade" and successfully prevented capital deployment into a non-profitable strategy, proving the robustness of the backtesting and regime-filtering logic.

---

## 📁 Repository Structure
- `src/data/`: Data loading and standardization.
- `src/features/`: Intraday feature engineering.
- `src/models/`: Regime detection and forecasting models.
- `src/execution/`: Alpaca broker integration and backtesting logic.
- `mlruns/`: MLflow experiment tracking database.
