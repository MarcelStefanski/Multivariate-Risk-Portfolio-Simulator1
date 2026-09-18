# Multivariate Risk and Portfolio Simulator

A Python-based financial risk and portfolio analytics project that models portfolio performance under uncertainty
using historical market data, Monte Carlo simulation, portfolio optimization, and stress testing.

## Overview

This project was developed to explore key concepts in quantitative finance, portfolio management, and risk analysis.

The simulator uses historical asset data to:

- Calculate daily returns
- Estimate covariance and correlation structures
- Analyze portfolio risk and return
- Generate Monte Carlo scenarios
- Calculate Value at Risk (VaR)
- Calculate Expected Shortfall (ES)
- Construct an Efficient Frontier
- Find Maximum Sharpe and Minimum Risk portfolios
- Perform stress testing under adverse market conditions


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

## Features

### Data Collection

- Historical market data downloaded using "yfinance"
- Daily adjusted closing prices

### Return Analysis

- Daily percentage returns
- Mean return calculations

### Risk Analysis

- Covariance matrix
- Correlation matrix
- Portfolio volatility

### Monte Carlo Simulation

- 10,000 simulated portfolio return scenarios
- Multivariate normal return generation

### Risk Measures

- Value at Risk (VaR)
- Expected Shortfall (ES)

### Portfolio Optimization

- Efficient Frontier generation
- Maximum Sharpe Ratio portfolio
- Minimum Risk portfolio

### Stress Testing

Portfolio performance evaluated under hypothetical market events:

- Moderate market crash
- Severe market crash
- Interest-rate shock
- Financial crisis scenario


## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- yfinance


## Example Results

### Maximum Sharpe Portfolio

- Annual Return: 13.40%
- Annual Risk: 12.89%

Portfolio Weights:

- SPY: 51.33%
- QQQ: 44.89%
- GLD: 2.78%
- TLT: 0.99%

### Minimum Risk Portfolio

- Annual Return: 6.97%
- Annual Risk: 9.38%

Portfolio Weights:

- SPY: 28.43%
- QQQ: 1.26%
- GLD: 31.36%
- TLT: 38.95%

## Project Structure

multivariate-risk-portfolio-simulator/

├── data_loader.py
├── return_calculator.py
├── covariance_analysis.py
├── portfolio_simulator.py
├── monte_carlo.py
├── simulation_plot.py
├── optimal_portfolios.py
├── stress_testing.py

├── results/
│   ├── monte_carlo_distribution.png
│   ├── efficient_frontier.png
│   └── optimal_portfolios.png

└── outputs/
