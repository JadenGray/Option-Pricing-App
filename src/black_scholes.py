import math
from scipy.stats import norm
from datetime import datetime


def _time_fraction(expiry, valuation_date=None):
    """
    Compute year fraction between valuation_date and expiry.
    """
    ref_date = valuation_date or datetime.utcnow()
    T = (expiry - ref_date).days / 365
    return max(T, 1e-6)


def black_scholes(S, K, expiry, r, sigma, option_type, valuation_date=None):
    """
    Calculate the Black–Scholes European option price.

    Parameters:
    - S: underlying spot price
    - K: strike price
    - expiry: expiration (datetime.date)
    - r: risk-free rate (annual)
    - sigma: volatility (annual)
    - option_type: 'call' or 'put'
    - valuation_date: datetime for pricing (defaults to now)

    Returns:
    - option price (float)
    """
    T = _time_fraction(expiry, valuation_date)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def implied_volatility(S, K, expiry, r, option_price, option_type, tol=1e-5, max_iter=100):
    """
    Compute implied volatility (annual) via Newton-Raphson.

    Returns volatility in percent.
    """
    T = _time_fraction(expiry, None)
    # Initial guess: Brenner-Subrahmanyam
    sigma = math.sqrt(max(2 * abs(math.log(S / K)) / T, 1e-6))

    for _ in range(max_iter):
        price = black_scholes(S, K, expiry, r, sigma, option_type)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        diff = price - option_price
        if abs(diff) < tol:
            return sigma * 100
        sigma -= diff / vega
    raise RuntimeError("Implied vol did not converge")
