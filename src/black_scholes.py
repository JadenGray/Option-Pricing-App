import math
from scipy.stats import norm
from datetime import datetime

# --- Pricing & Greeks ---

def _time_fraction(expiry, valuation_date):
    T = (expiry - (valuation_date or datetime.utcnow())).days / 365
    return max(T, 1e-6)

def black_scholes_metric(S, K, expiry, r, sigma, option_type, quantity, valuation_date=None):
    """Universal BS function for Price, Delta, Gamma, Vega, Theta, Rho"""
    T = _time_fraction(expiry, valuation_date)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    lookup = {
        ('call', 'Price'):      S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2),
        ('put',  'Price'):      K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1),
        ('call', 'Delta'):      norm.cdf(d1),
        ('put',  'Delta'):      norm.cdf(d1) - 1,
        ('call', 'Gamma'):      norm.pdf(d1)/(S*sigma*math.sqrt(T)),
        ('put',  'Gamma'):      norm.pdf(d1)/(S*sigma*math.sqrt(T)),
        ('call', 'Vega'):       S * norm.pdf(d1) * math.sqrt(T),
        ('put',  'Vega'):       S * norm.pdf(d1) * math.sqrt(T),
        ('call', 'Theta'):     ( -S*norm.pdf(d1)*sigma/(2*math.sqrt(T))
                                 - r*K*math.exp(-r*T)*norm.cdf(d2) ),
        ('put',  'Theta'):     ( -S*norm.pdf(d1)*sigma/(2*math.sqrt(T))
                                 + r*K*math.exp(-r*T)*norm.cdf(-d2) ),
        ('call', 'Rho'):        K*T*math.exp(-r*T)*norm.cdf(d2),
        ('put',  'Rho'):       -K*T*math.exp(-r*T)*norm.cdf(-d2),
    }
    return lookup[(option_type, quantity)]

# --- Implied Volatility via Newton-Raphson ---

def implied_volatility(S, K, expiry, r, option_price, option_type, tol=1e-5, max_iter=100):
    """Return implied vol (%) for a market option price."""
    T = _time_fraction(expiry, None)
    sigma = math.sqrt(2*abs(math.log(S/K))/T)
    for _ in range(max_iter):
        price = black_scholes_metric(S, K, expiry, r, sigma, option_type, 'Price')
        d1 = (math.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
        vega = S*norm.pdf(d1)*math.sqrt(T)
        diff = price - option_price
        if abs(diff) < tol:
            return sigma*100
        sigma -= diff/vega
    raise RuntimeError("Implied vol did not converge")