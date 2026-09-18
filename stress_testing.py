import numpy as np

# Portfolio weights
weights = np.array([
    0.40,  # SPY
    0.30,  # QQQ
    0.15,  # GLD
    0.15   # TLT
])

# Stress scenarios
scenarios = {
    "Moderate Crash": [
        -0.10,  # SPY
        -0.15,  # QQQ
         0.02,  # GLD
         0.05   # TLT
    ],

    "Severe Crash": [
        -0.25,
        -0.35,
         0.10,
         0.15
    ],

    "Interest Rate Shock": [
        -0.05,
        -0.08,
         0.03,
        -0.20
    ],

    "Financial Crisis": [
        -0.40,
        -0.50,
         0.15,
         0.20
    ]
}

print("STRESS TEST RESULTS")
print("-" * 40)

for name, returns in scenarios.items():

    scenario_returns = np.array(returns)

    portfolio_return = np.dot(
        weights,
        scenario_returns
    )

    print(f"\n{name}")
    print(f"Portfolio Return: {portfolio_return:.2%}")