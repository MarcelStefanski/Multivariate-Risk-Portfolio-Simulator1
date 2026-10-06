# Multivariate Risk and Portfolio Simulator

A Python-based financial risk and portfolio analytics project that models portfolio performance under uncertainty using historical market data, Monte Carlo simulation, portfolio optimization, stress testing, and performance analysis.

## Overview

This project was developed to explore key concepts in quantitative finance, portfolio management, and risk analysis.

The simulator uses historical asset data to:

- Calculate daily returns
- Estimate covariance and correlation structures
- Analyze portfolio risk and return
- Generate Monte Carlo scenarios
- Calculate Value at Risk (VaR)
- Calculate Expected Shortfall (ES)
- Compare historical and Monte Carlo VaR and Expected Shortfall
- Sample feasible portfolio risk/return combinations
- Find sample-best Sharpe and minimum-risk portfolios
- Perform stress testing under adverse market conditions
- Measure risk contributions by asset
- Perform Principal Component Analysis (PCA)
- Backtest historical portfolio performance
- Compare portfolio performance against a benchmark

---

## Assets

The project currently uses:

- SPY (S&P 500 ETF)
- QQQ (Nasdaq-100 ETF)
- GLD (Gold ETF)
- TLT (20+ Year Treasury Bond ETF)

These assets provide exposure to:

- Equities
- Technology stocks
- Commodities
- Fixed income

---

## Features

### Data Collection

- Historical market data downloaded using `yfinance`
- Daily adjusted closing prices

### Return Analysis

- Daily percentage returns
- Mean return calculations

### Risk Analysis

- Covariance matrix
- Correlation matrix
- Portfolio volatility
- Risk contribution analysis

### Monte Carlo Simulation

- 10,000 simulated portfolio return scenarios
- Multivariate normal return generation

### Risk Measures

- Value at Risk (VaR)
- Expected Shortfall (ES)
- Maximum Drawdown
- Sortino Ratio (0% target return)

### Portfolio Sampling

- Fixed-seed random sampling of 5,000 long-only portfolio weights
- Best sampled Maximum Sharpe Ratio and Minimum Risk portfolios

### Stress Testing

Portfolio performance evaluated under hypothetical market events:

- Moderate market crash
- Severe market crash
- Interest-rate shock
- Financial crisis scenario

### Principal Component Analysis (PCA)

- Risk decomposition using PCA
- Explained variance by principal component
- Graphical visualization of dominant risk factors

### Historical Backtesting

- Daily, frictionless baseline compared with configurable monthly or quarterly rebalancing
- Configurable transaction costs, portfolio growth, annualized return and volatility, maximum drawdown, and turnover
- Monthly vs quarterly cost-aware comparison using the same cost assumption
- Monthly and quarterly results compared across transaction-cost assumptions
- Fixed-weight holdout backtest compared with SPY
- Fama-French three-factor regression for portfolio alpha and factor exposure

### Benchmark Comparison

- Monthly or quarterly cost-aware portfolio vs buy-and-hold SPY over the same period
- Return, volatility, drawdown, and portfolio turnover comparison
- Comparative growth visualization

### Factor Analysis

- Estimates portfolio exposure to the market, size, and value factors
- Reports annualized alpha and R-squared using daily returns after configured transaction costs
- Uses daily factor data from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

### Performance Metrics

- Annualized arithmetic return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- yfinance

---

## Key Results

The portfolio results below are saved example outputs from an earlier 5,000-sample run. The optimizer now uses a fixed random seed, so its results may differ from this snapshot.

### Best Sampled Maximum Sharpe Portfolio

- Annualized Arithmetic Return: 13.40%
- Annual Risk: 12.89%

Portfolio Weights:

- SPY: 51.33%
- QQQ: 44.89%
- GLD: 2.78%
- TLT: 0.99%

### Best Sampled Minimum Risk Portfolio

- Annualized Arithmetic Return: 6.97%
- Annual Risk: 9.38%

Portfolio Weights:

- SPY: 28.43%
- QQQ: 1.26%
- GLD: 31.36%
- TLT: 38.95%

### Portfolio Performance

- Annualized Arithmetic Return: 11.24%
- Annual Volatility: 11.22%
- Sharpe Ratio (0% risk-free rate): 1.00
- Maximum Drawdown: -23.10%

### Benchmark Comparison

Portfolio:

- Annualized Arithmetic Return: 11.24%
- Annual Volatility: 11.22%

SPY Benchmark:

- Annualized Arithmetic Return: 13.81%
- Annual Volatility: 17.62%

The diversified portfolio generated lower returns than SPY while reducing volatility by approximately 36%.

### Risk Contribution Analysis

Risk Contribution by Asset:

- SPY: 30.77%
- QQQ: 46.72%
- GLD: 17.86%
- TLT: 4.65%

This highlights that QQQ is the primary driver of portfolio risk despite representing only 30% of portfolio weight.

### Stress Testing Results

| Scenario | Portfolio Return |
|-----------|-----------------|
| Moderate Crash | -7.45% |
| Severe Crash | -16.75% |
| Interest Rate Shock | -6.95% |
| Financial Crisis | -25.75% |

---


## Future Improvements

Potential extensions include:

- Interactive dashboard
- Copula-based dependence modeling
- Advanced risk attribution

---

## Learning Outcomes

This project demonstrates practical applications of:

- Portfolio Theory
- Quantitative Risk Management
- Monte Carlo Methods
- Statistical Modelling
- Principal Component Analysis
- Financial Data Analysis
- Portfolio Optimization
- Performance Evaluation
- Python Programming
