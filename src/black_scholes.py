import math
from scipy.stats import norm

# taken from option app code
def black_scholes(S, K, T, r, sigma, option_type, quantity):
    """
    Calculate the Black-Scholes price and greeks for a European call or put option.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if quantity == "Price":
        # use black scholes formula
        if option_type == "call":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    # Greek formulae taken from https://www.quantpie.co.uk/bsm_formula/bs_summary.php
    elif quantity == "Delta":
        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
            return norm.cdf(d1) - 1
    elif quantity == "Gamma":
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    elif quantity == "Vega":
        return S * norm.pdf(d1) * math.sqrt(T)
    elif quantity == "Theta":
        if option_type == "call":
            return (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                    - r * K * math.exp(-r * T) * norm.cdf(d2))
        elif option_type == "put":
            return (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                    + r * K * math.exp(-r * T) * norm.cdf(-d2))
    elif quantity == "Rho":
        if option_type == "call":
            return K * T * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return -K * T * math.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid Quantity. Choose from Price, Delta, Gamma, Vega, Theta, Rho.")

def implied_volatility(S, K, T, r, option_price, option_type, tol=1e-5, max_iter=100):
    """
    Calculate the implied volatility for a European call or put option using the Newton-Raphson method.
    Details on the method can be found at https://medium.com/hypervolatility/extracting-implied-volatility-newton-raphson-secant-and-bisection-approaches-fae83c779e56
    Initial guess constructed on a logarithmic approximation: https://quant.stackexchange.com/questions/58634/newtons-algorithm-for-implied-volatility
    """
    # Initial guess for sigma
    numerator = math.sqrt(2 * math.log((S * math.exp(r * T)) / K))
    denominator = math.sqrt(T)
    sigma = numerator / denominator if denominator > 0 else 0.2  # Fallback if T = 0
    # Iterate the newton raphson formula
    for _ in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type, "Price")
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        price_diff = price - option_price
        if abs(price_diff) < tol:
            return sigma * 100
        sigma -= price_diff / vega
    raise ValueError("Implied volatility did not converge.")