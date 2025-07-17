# packages
import math
from scipy.stats import norm
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# setup boxes to make app look nicer
st.markdown("""
        <style>
        .green-box {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            padding: 10px;
            border-radius: 5px;
            font-size: 20px;
            text-align: center;
        }
        .red-box {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            padding: 10px;
            border-radius: 5px;
            font-size: 20px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

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

def portfolio_metrics(portfolio, rate, spot_price, volatility, metric="Price"):
    """
    Compute the portfolio-wide metric by summing individual option metrics.
    
    Parameters:
    - portfolio: List of options, where each option is a dictionary.
    - rate: Risk-free rate (r), applied globally to all options.
    - spot_price: Current stock price (S), applied globally to all options.
    - volatility: Volatility (σ), applied globally to all options.
    - metric: The desired metric to compute (default is "Price").
    
    Returns:
    - total_metric: The total portfolio metric value.
    """
    total_metric = 0
    # iterate through each option in the portfolio
    for option in portfolio:
        metric_value = black_scholes(
            spot_price,                # Spot price (S)
            option["strike"],          # Strike price (K)
            option["time_to_maturity"],# Time to maturity (T)
            rate,                      # Risk-free rate (r)
            volatility,                # Volatility (σ)
            option["type"],            # Option type ("call" or "put")
            metric                     # Metric to compute (e.g., "Price")
        )
        # add the values
        total_metric += option["quantity"] * metric_value
    return total_metric

# Make the streamlit app work
# Initialize session state for navigation
if "section" not in st.session_state:
    st.session_state.section = "Home"

# Navigation buttons
def navigate_to(section):
    st.session_state.section = section

st.sidebar.header("Navigation")
st.sidebar.button("Option Calculations", on_click=navigate_to, args=("Option Calculations",))
st.sidebar.button("Heatmaps", on_click=navigate_to, args=("Heatmaps",))
st.sidebar.button("Portfolio Analysis", on_click=navigate_to, args=("Portfolio Analysis",))
st.sidebar.write("---")  # Divider line for better separation

# Make the Opiton Heatmap page and the sidebar
def option_calculations():
    # titles
    st.title("Option Calculations")
    st.write("Calculate different quantities related to options using the Black-Scholes model.")
    
    # parameter sidebar
    st.sidebar.header("Parameters")
    with st.sidebar:
        calc_type = st.selectbox("Select Quantity to Calculate", ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "Implied Volatility"])
        T = st.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
        K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
        r = st.slider("Risk-Free Rate (r)", min_value=0.0, max_value=1.0, value=0.05)
        S = st.number_input("Stock Price (S)", min_value=0.01, value=100.0, step=1.0)
    
    # implied vol sidebar addition      
    if calc_type == "Implied Volatility":
        with st.sidebar:
            option_type = st.selectbox("Option Type", ["call", "put"])
            purchase_price = st.number_input("Purchase Price", min_value=0.01, value=10.0, step=0.1)
            
        implied_vol = implied_volatility(S, K, T, r, purchase_price, option_type)
        with st.container():
            st.markdown(f"<div class='green-box'> Implied Volatility: {implied_vol:.2f}% </div>", unsafe_allow_html=True)
    
    # additions otherwise
    else:
        with st.sidebar:
            sigma = st.slider("Actual Volatility (σ)", min_value=0.0, max_value=1.0, value=0.2)
            
        call_quantity = black_scholes(S, K, T, r, sigma, "call", calc_type)
        put_quantity = black_scholes(S, K, T, r, sigma, "put", calc_type)

        col1, col2 = st.columns(2)
        if calc_type == "Price":
            with col1:
                st.markdown(f"<div class='green-box'> CALL: ${call_quantity:.2f}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='red-box'> PUT: ${put_quantity:.2f}</div>", unsafe_allow_html=True)
        else:
            with col1:
                st.markdown(f"<div class='green-box'> CALL: {call_quantity:.2f}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='red-box'> PUT: {put_quantity:.2f}</div>", unsafe_allow_html=True)
    

# function to make the heatmaps
def heatmaps():
    st.title("Heatmaps")
    st.write("Generate heatmaps for option prices or Greeks.")
    # Heatmap Parameters Section
    st.sidebar.header("Heatmap Parameters")
    with st.sidebar:
        heatmap_quantity = st.sidebar.selectbox("Select Quantity to Visualize", ["PnL", "Delta", "Gamma", "Vega", "Theta", "Rho"])
        option_type = st.selectbox("Option Type", ["call", "put"])
        # Show purchase price input only if PnL is selected
        if heatmap_quantity == "PnL":
            purchase_price = st.number_input("Purchase Price", min_value=0.01, value=10.0, step=0.1)
        else:
            purchase_price = 10.0  # Set to None if not applicable
        T = st.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
        K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
        r = st.slider("Risk-Free Rate (r)", min_value=0.0, max_value=1.0, value=0.05)
        
        # Spot price range
        spot_min = st.number_input("Min Spot Price (S)", min_value=0.01, value=50.0, step=1.0)
        spot_max = st.number_input("Max Spot Price (S)", min_value=spot_min + 0.01, value=150.0, step=1.0)
        if spot_max <= spot_min:
            st.error("Max Spot Price must be greater than Min Spot Price.")
            spot_max = spot_min + 1  # Auto-correct to avoid breaking

        # Volatility range
        vol_min = st.slider("Min Volatility (σ)", min_value=0.01, max_value=0.99, value=0.01, step=0.01)
        if vol_min == 0.99:
            vol_max = 1.0
        else:
            vol_max = st.slider("Max Volatility (σ)", min_value=vol_min + 0.01, max_value=1.0, value=1.0, step=0.01)
            if vol_max <= vol_min:
                st.error("Max Volatility must be greater than Min Volatility.")
                vol_max = vol_min + 0.01  # Auto-correct to avoid breaking
        
    # Calculate PnL Heatmap
    spot_prices = np.linspace(spot_min, spot_max, 10)
    volatilities = np.linspace(vol_min, vol_max, 10)

    # X and Y values
    X, Y = np.meshgrid(spot_prices, volatilities)

    # Initialize an empty grid for the chosen metric
    metric_values = np.zeros_like(X)

    # Loop through the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # naming issue fix - want to calculate pnl instead of price
            if heatmap_quantity == "PnL":
                option_price = black_scholes(X[i, j], K, T, r, Y[i, j], option_type, "Price")
                # price difference to get pnl
                metric_values[i, j] = option_price - purchase_price
            else:
                # can just compute the greeks using earlier funciton
                metric_values[i, j] = black_scholes(X[i, j], K, T, r, Y[i, j], option_type, heatmap_quantity)


    # Display Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    # Adjust colormap and normalization based on the metric
    if heatmap_quantity == "PnL":
        # make the colour gradient centered at 0
        color_norm = TwoSlopeNorm(vmin=min(-0.01, np.min(metric_values)), vcenter=0, vmax=max(0.01, np.max(metric_values)))
        cmap = "RdYlGn"
        cbar_label = "PnL"
    else:
        color_norm = None  # No need for centered colormap for Greeks
        cmap = "coolwarm"
        cbar_label = heatmap_quantity

    # Plot the heatmap
    st.subheader(f"{heatmap_quantity} Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        metric_values,
        annot=True,
        fmt=".2f",
        xticklabels=np.round(spot_prices, 2),
        yticklabels=np.round(volatilities, 2),
        cmap=cmap,
        norm=color_norm,
        ax=ax,
        cbar_kws={'label': cbar_label}
    )
    
    if heatmap_quantity == "PnL": # colours are tricker for this quantity  
        # Adjust colorbar ticks and labels
        colorbar = ax.collections[0].colorbar
        num_ticks = 11  # Total number of ticks

        # Generate evenly spaced ticks on the normalized scale
        ticks = np.linspace(0, 1, num_ticks)  # Evenly spaced ticks from 0 to 1
        scaled_ticks = color_norm.inverse(ticks)  # Map normalized ticks back to data values
        colorbar.set_ticks(scaled_ticks)
        colorbar.set_ticklabels([f"{tick:.2f}" for tick in scaled_ticks])

    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Volatility (σ)")
    st.pyplot(fig)
    
    
# Similar to previous function with added portfolio abilities
def portfolio_analysis():
    st.title("Portfolio Analysis")
    st.write("Analyze portfolios of multiple options.")

    # Allow users to choose the metric to visualize
    heatmap_quantity = st.sidebar.selectbox("Select Quantity to Visualize", ["PnL", "Delta", "Gamma", "Vega", "Theta", "Rho"])

    st.sidebar.header("Portfolio Setup")
    with st.sidebar:
        r = st.slider("Risk-Free Rate (r)", min_value=0.0, max_value=1.0, value=0.05)
        sigma = st.slider("Actual Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2)
        S = st.number_input("Stock Price (S)", min_value=0.01, value=100.0, step=1.0)

        # Spot price range
        spot_min = st.number_input("Min Spot Price (S)", min_value=0.01, value=50.0, step=1.0)
        spot_max = st.number_input("Max Spot Price (S)", min_value=spot_min + 0.01, value=150.0, step=1.0)
        if spot_max <= spot_min:
            st.error("Max Spot Price must be greater than Min Spot Price.")
            spot_max = spot_min + 1  # Auto-correct to avoid breaking

        # Volatility range
        vol_min = st.slider("Min Volatility (σ)", min_value=0.01, max_value=0.99, value=0.01, step=0.01)
        vol_max = st.slider("Max Volatility (σ)", min_value=vol_min + 0.01, max_value=1.0, value=1.0, step=0.01)
        if vol_max <= vol_min:
            st.error("Max Volatility must be greater than Min Volatility.")
            vol_max = vol_min + 0.01  # Auto-correct to avoid breaking (bug fix)

    # number of options in the portfolio
    num_options = st.sidebar.number_input("Number of Options in Portfolio", min_value=1, value=1, step=1)

    portfolio = []
    # build the option portfolio
    for i in range(num_options):
        with st.sidebar.expander(f"Option {i+1}"):
            option_type = st.selectbox(f"Type of Option {i+1}", ["call", "put"], key=f"type_{i}")
            K = st.number_input(f"Strike Price (K) for Option {i+1}", min_value=0.01, value=100.0, step=1.0, key=f"K_{i}")
            T = st.number_input(f"Time to Maturity (T) for Option {i+1}", min_value=0.01, value=1.0, step=0.01, key=f"T_{i}")
            quantity = st.number_input(f"Quantity for Option {i+1}", min_value=-100, max_value=100, value=1, key=f"quantity_{i}")
            purchase_price = st.number_input(f"Purchase Price for Option {i+1}", min_value=0.01, value=10.0, step=1.0, key=f"purchase_price_{i}")

            portfolio.append({
                "type": option_type,
                "strike": K,
                "time_to_maturity": T,
                "quantity": quantity,
                "purchase_price": purchase_price
            })

    # Generate heatmap data
    spot_prices = np.linspace(spot_min, spot_max, 10)
    volatilities = np.linspace(vol_min, vol_max, 10)

    X, Y = np.meshgrid(spot_prices, volatilities)
    portfolio_values = np.zeros_like(X)

    # Fill in the data using the portfolio
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Go through the spot prices and volatilities and each option
            simulated_portfolio = [
                {**option, "spot_price": X[i, j], "volatility": Y[i, j]} for option in portfolio
            ]
            # Compute the selected metric
            # as in previous function, pnl is dealt with accordingly
            if heatmap_quantity == "PnL":
                portfolio_values[i, j] = (
                    portfolio_metrics(simulated_portfolio, rate=r, spot_price=X[i, j], volatility=Y[i, j], metric="Price") -
                    sum(option["purchase_price"] * option["quantity"] for option in portfolio)
                )
            else:
                portfolio_values[i, j] = portfolio_metrics(
                    simulated_portfolio, rate=r, spot_price=X[i, j], volatility=Y[i, j], metric=heatmap_quantity
                )

    # Display heatmap
    st.subheader(f"{heatmap_quantity} Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Adjust colormap and normalization (same as previous function)
    if heatmap_quantity == "PnL":
        color_norm = TwoSlopeNorm(vmin=min(-0.01, np.min(portfolio_values)), vcenter=0, vmax=max(0.01, np.max(portfolio_values)))
        cmap = "RdYlGn"
        cbar_label = "Portfolio PnL"
    else:
        color_norm = None  # No need for centered colormap for Greeks
        cmap = "coolwarm"
        cbar_label = heatmap_quantity

    # Display heatmap
    sns.heatmap(
        portfolio_values,
        annot=True,
        fmt=".2f",
        xticklabels=np.round(spot_prices, 2),
        yticklabels=np.round(volatilities, 2),
        cmap=cmap,
        norm=color_norm,
        ax=ax,
        cbar_kws={"label": cbar_label}
    )

    if heatmap_quantity == "PnL":
        # Adjust colorbar ticks and labels for PnL
        colorbar = ax.collections[0].colorbar
        num_ticks = 11  # Total number of ticks
        ticks = np.linspace(0, 1, num_ticks)
        scaled_ticks = color_norm.inverse(ticks)
        colorbar.set_ticks(scaled_ticks)
        colorbar.set_ticklabels([f"{tick:.2f}" for tick in scaled_ticks])

    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Volatility (σ)")
    st.pyplot(fig)
    # Heatmap done

    # PnL plot below heatmap
    with st.sidebar: # Add sidebar options for the plot
        st.write("---")
        st.header("PnL Plot Parameters")
        # Stock price range for PnL curve
        stock_price_min = st.number_input("Min Stock Price (S)", min_value=0.01, value=50.0, step=1.0)
        stock_price_max = st.number_input("Max Stock Price (S)", min_value=stock_price_min + 0.01, value=150.0, step=1.0)
        if stock_price_max <= stock_price_min:
            st.error("Max Stock Price must be greater than Min Stock Price.")
            stock_price_max = stock_price_min + 1
            
    # Stock price range for curve
    stock_prices = np.linspace(stock_price_min, stock_price_max, 100)

    # Calculate PnL for the portfolio across stock prices
    portfolio_pnl = [
        portfolio_metrics(portfolio, rate=r, spot_price=S, volatility=sigma, metric="Price") - 
        sum(option["purchase_price"] * option["quantity"] for option in portfolio)
        for S in stock_prices
    ]

    # Plot PnL curve
    st.subheader("Portfolio PnL vs Stock Price")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(stock_prices, portfolio_pnl, label="Portfolio PnL", color="blue")
    ax.axhline(0, color="black", linestyle="--", label="Break-Even Line")
    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("PnL")
    ax.set_title("Portfolio PnL vs Stock Price")
    ax.legend()
    st.pyplot(fig)
        
        
# Make the app work
# Render the selected section
if st.session_state.section == "Option Calculations":
    option_calculations()
elif st.session_state.section == "Heatmaps":
    heatmaps()
elif st.session_state.section == "Portfolio Analysis":
    portfolio_analysis()
else:
    st.title("Welcome")
    st.write("Use the navigation buttons in the sidebar to explore the app.")