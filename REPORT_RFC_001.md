# RFC 001: Regime-Aware Intraday Forecasting System
**Status**: Research Concluded (Negative Deployment Recommendation)
**Author**: Antigravity Quant Research
**Date**: 2026-01-01

## 1. Executive Summary
This document outlines the architecture, methodology, and results of a production-grade machine learning pipeline designed to trade financial assets (SPY, XAUUSD, EURUSD) using regime-conditioned expert models. While the technical infrastructure achieved all design goals for reliability and MLOps integration, empirical results across three asset classes demonstrated that directional predictability on intraday timeframes is insufficient to overcome modern transaction costs.

## 2. System Architecture & Pipeline
The system is built as a modular end-to-end ML pipeline with the following stages:

### 2.1 Data Ingestion Layer
- **Multi-Source Support**: Integration with `yfinance` for public equities/forex and custom CSV loaders for proprietary high-frequency data (semicolon-separated, specific date formats).
- **Timeframe Alignment**: Logical synchronization between 5m (execution), 15m (decision), and 1h (strategic) bars.

### 2.2 Feature Engineering
We employed a "Strict Lookback" policy to prevent data leakage:
- **Momentum**: ROC (Rate of Change) over 4, 12, and 48 bars.
- **Volatility**: Rolling standard deviation and Relative Volatility (Short/Long ratio).
- **Trend**: Linear regression slopes and price-position relative to moving averages.
- **Normalization**: ATR-normalized ranges to ensure stationarity in feature distributions across different volatility regimes.

### 2.3 Modeling: The Regime-Aware Ensemble
The core innovation is the separation of "Environment Detection" from "Return Forecasting."

#### Stage 1: Regime Clustering (Unsupervised/Heuristic)
We classify the market into three distinct states:
1. **p_high_vol**: State of expansion, usually event-driven.
2. **p_trending**: State of persistent directional flow.
3. **p_ranging**: State of mean-reversion and noise.
The ensemble uses a combination of an **HMM (Hidden Markov Model)** and rolling volatility percentiles to assign a probability vector to each bar.

#### Stage 2: Expert Forecasting (Supervised)
- **Expert Models**: Individual LightGBM regressors trained *only* on data from specific regimes.
- **Inference**: A weighted average prediction: `Forecast = Sum(Regime_Prob_i * Expert_Pred_i)`.

### 2.4 Experiment Tracking & MLOps
- **MLflow Integration**: Every walk-forward step (Weekly/Monthly) is logged with hyperparameters, MSE metrics, and model artifacts.
- **Walk-Forward Validation**: A sliding window approach (6-week train / 1-week test) was used to simulate real-world model degradation and retraining.

---

## 3. Backtesting Results & Observations

### 3.1 Asset: SPY (Baseline)
- **Timeframe**: 15m
- **Period**: 60 Days (Initial Smoke Test)
- **Metrics (Naive Baseline)**:
  - **Total Return**: 59.22%
  - **Annualized Sharpe**: 16.65
  - **Max Drawdown**: -1.65%
  - **Win Rate**: 58.97%
  - **Total Trades**: 1165
- **Observation**: This initial naive test was used to verify pipeline connectivity. The "too good to be true" Sharpe (16.65) served as a vital research indicator of **look-ahead bias** and **absent transaction costs**, prompting the transition to the more rigorous XAUUSD walk-forward framework.
- **Final Verdict**: Baseline verified infrastructure, but results were rejected for live-trading consideration due to lack of slippage/cost modeling.

### 3.2 Asset: XAUUSD (Gold)
- **Timeframe**: 5m / 15m
- **Strategy**: Sniper (High-Vol only)
- **Stats**: Total Return: -19.8% (v1) -> ~0.0% (v2 - Pivot).
- **Observations**: The "Microstructure Drag" in Gold is extreme. The asset exhibits high "flicker" (mean reversion in small timeframes) that destroys momentum-based ML signals.
- **Equity Curve**: Characterized by sharp drawdowns during ranging periods and flat/recovery periods during news events.

### 3.3 Asset: EURUSD (Forex)
- **Timeframe**: 1H
- **Stats**: Total Return: -1.08%
- **Observations**: While spread is lower, directional predictability (Correlation: 0.009) is effectively zero. The market is efficiently priced for the features used.

---

## 4. Key Metrics
- **Sharpe Ratio**: Consistently negative across intraday assets (-0.4 to -0.9).
- **Win Rate**: Hovered between 47% and 49%, which is insufficient for "Trend Following" without a massive Reward/Risk ratio.
- **Transaction Cost Impact**: Calculated at ~1bps (XAU) and 0.5bps (EURUSD). This single variable was the primary driver of negative PnL.

---

## 5. Lessons Learned & Observations
1. **The "Cost Barrier"**: In intraday trading, your first competitor isn't the other traders—it's the broker and the spread. Any edge smaller than the round-trip cost is mathematically irrelevant.
2. **Regime Success**: The HMM + Volatility ensemble was highly effective at identifying *state*. It correctly predicted *high risk* periods.
3. **Forecast Fragility**: Directional forecasting on price alone (OHLC) is a "weak signal" task. In 2026 markets, this signal is largely exploited by HFTs, leaving little for retail-available features.
4. **Research Value**: The project was a success in its primary goal: **Developing a valid research pipeline that kills bad ideas before they cost real capital.**

## 6. Final Recommendation
**Do not deploy.** The directional alpha is too thin. Future work should pivot toward **Volatility Arbitrage** or move to **Daily/Weekly timeframes** where the signal-to-noise ratio is significantly higher.
