from pathlib import Path

import numpy as np
import yfinance as yf


TICKERS = ["SPY", "QQQ", "GLD", "TLT"]
WEIGHTS = np.array([0.4, 0.3, 0.15, 0.15])
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
NUM_SIMULATIONS = 10_000
CONFIDENCE_LEVELS = (0.95, 0.99)

data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
prices = data["Close"].loc[:, TICKERS]
asset_returns = prices.pct_change().dropna()
historical_portfolio_returns = asset_returns @ WEIGHTS

# Simulate daily asset returns from their historical multivariate normal model.
rng = np.random.default_rng(42)
simulated_asset_returns = rng.multivariate_normal(
    asset_returns.mean().to_numpy(),
    asset_returns.cov().to_numpy(),
    size=NUM_SIMULATIONS,
)
simulated_portfolio_returns = simulated_asset_returns @ WEIGHTS

# Express both VaR and Expected Shortfall as positive loss percentages.
historical_losses = -historical_portfolio_returns.to_numpy()
simulated_losses = -simulated_portfolio_returns

lines = [
    "HISTORICAL VS MONTE CARLO PORTFOLIO RISK",
    "One-day loss estimates; VaR and Expected Shortfall shown as positive losses.",
    f"Monte Carlo simulations: {NUM_SIMULATIONS:,}",
    "Monte Carlo model: multivariate normal returns using historical mean and covariance.",
    "",
    "Confidence       Historical VaR   Historical ES   Monte Carlo VaR   Monte Carlo ES",
]

for confidence in CONFIDENCE_LEVELS:
    historical_var = np.quantile(historical_losses, confidence)
    historical_es = historical_losses[historical_losses >= historical_var].mean()
    monte_carlo_var = np.quantile(simulated_losses, confidence)
    monte_carlo_es = simulated_losses[simulated_losses >= monte_carlo_var].mean()

    lines.append(
        f"{confidence:>8.0%}"
        f"          {historical_var:>8.2%}"
        f"         {historical_es:>8.2%}"
        f"           {monte_carlo_var:>8.2%}"
        f"          {monte_carlo_es:>8.2%}"
    )

result = "\n".join(lines)
print(result)

output_path = Path("results/historical_vs_monte_carlo_var.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(result + "\n", encoding="utf-8")
