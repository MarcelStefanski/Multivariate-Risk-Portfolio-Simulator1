from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from backtesting import (
    END_DATE,
    REBALANCE_FREQUENCY,
    START_DATE,
    TARGET_WEIGHTS,
    TICKERS,
    TRANSACTION_COST_BPS,
    TRADING_DAYS,
    cost_aware_backtest,
)


FACTOR_DATA_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FACTOR_NAMES = ["Mkt-RF", "SMB", "HML"]


def load_factor_data():
    """Download daily Fama-French factors and convert percentages to decimals."""
    with urlopen(FACTOR_DATA_URL) as response:
        with ZipFile(BytesIO(response.read())) as archive:
            csv_text = archive.read(archive.namelist()[0]).decode("latin-1")

    factors = pd.read_csv(StringIO(csv_text), skiprows=3)
    factors = factors.iloc[:, :5]
    factors.columns = ["Date", *FACTOR_NAMES, "RF"]
    factors["Date"] = factors["Date"].astype(str)
    factors = factors[factors["Date"].str.fullmatch(r"\d{8}")].copy()
    factors["Date"] = pd.to_datetime(factors["Date"], format="%Y%m%d")
    factors = factors.set_index("Date").astype(float) / 100

    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)
    return factors.loc[(factors.index >= start) & (factors.index < end)]


def fit_factor_model(portfolio_returns, factors):
    """Fit excess portfolio returns against the three daily factor returns."""
    data = pd.concat([portfolio_returns, factors], axis=1).dropna()
    if len(data) < len(FACTOR_NAMES) + 2:
        raise ValueError("Not enough overlapping daily returns to fit the model.")

    factor_matrix = data[FACTOR_NAMES].to_numpy()
    design_matrix = np.column_stack([np.ones(len(data)), factor_matrix])
    excess_returns = (data["Portfolio"] - data["RF"]).to_numpy()
    coefficients, *_ = np.linalg.lstsq(design_matrix, excess_returns, rcond=None)

    fitted_returns = design_matrix @ coefficients
    residuals = excess_returns - fitted_returns
    total_variation = np.sum((excess_returns - excess_returns.mean()) ** 2)
    r_squared = 1 - np.sum(residuals**2) / total_variation

    return {
        "alpha": coefficients[0] * TRADING_DAYS,
        "betas": dict(zip(FACTOR_NAMES, coefficients[1:])),
        "r_squared": r_squared,
        "observations": len(data),
        "start_date": data.index[0].date(),
        "end_date": data.index[-1].date(),
    }


def main():
    prices = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Close"]
    asset_returns = prices.loc[:, TICKERS].pct_change().dropna()
    portfolio_daily_returns, _, _ = cost_aware_backtest(
        asset_returns,
        TARGET_WEIGHTS,
        REBALANCE_FREQUENCY,
        TRANSACTION_COST_BPS,
    )
    portfolio_returns = pd.Series(
        portfolio_daily_returns, index=asset_returns.index, name="Portfolio"
    )

    model = fit_factor_model(portfolio_returns, load_factor_data())
    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    result = f"""FAMA-FRENCH THREE-FACTOR ANALYSIS
Portfolio weights: {weight_summary}
Rebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded
Aligned period: {model['start_date']} to {model['end_date']}
Daily observations: {model['observations']}

Annualized alpha: {model['alpha']:.2%}
Market beta (Mkt-RF): {model['betas']['Mkt-RF']:.3f}
Size beta (SMB): {model['betas']['SMB']:.3f}
Value beta (HML): {model['betas']['HML']:.3f}
R-squared: {model['r_squared']:.2%}

The model explains portfolio excess returns using the market, size, and value factors.
Gold and bond risks may not be captured by these U.S. equity factors."""
    print(result)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "factor_analysis.txt").write_text(result + "\n", encoding="utf-8")

    labels = ["Market (Mkt-RF)", "Size (SMB)", "Value (HML)"]
    betas = [model["betas"][name] for name in FACTOR_NAMES]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, betas, color=["steelblue", "darkorange", "seagreen"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Portfolio Exposure to Fama-French Factors")
    plt.ylabel("Estimated beta")
    plt.tight_layout()
    plt.savefig(results_dir / "factor_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
