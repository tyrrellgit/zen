import streamlit as st

from openbb import obb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from skfolio.preprocessing import prices_to_returns

from skfolio import RiskMeasure
from skfolio.cluster import HierarchicalClustering
from skfolio.optimization import (
    HierarchicalRiskParity, 
    HierarchicalEqualRiskContribution,
)

from src.comparator import Comparator

# Streamlit app title
st.title("Portfolio Rebalancing Tool")
st.markdown(
"""
    Portfolio rebalancing tool using Hierarchical clustering techniques for Risk allocation.
    
    - HRP : Hierarchical Risk Parity
    - HERC : Hierarchical Equal Risk Contribtuion
    
    These are modern approaches to portfolio theory that amount to clustering similar assets in terms of their
    returns covariances and allocating budget in a top-down manner such that risk is distributed accross clusters.

    ---

    Two benchmark portfolios are used to compare performance:

    1) Equal Weighting
    2) S&P500 Buy & Hold

    Equal Weigting is as it sounds; we take the same assets as the Hierarchical portfolios but we simply weight each asset
    equally (i.e. identical capital allocation).

    The remaining benchmark is the Buy & Hold strategy for S&P500, this is again as simple as it sounds; buy S&P500 index
    at period start and hold for same period. 

    ---
""" 
)

# User inputs for tickers and date range
tickers = st.text_input("Enter tickers separated by commas", "AAPL, AMZN, MSFT,  GOOGL, BABA, NVDA, TSLA")
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))

if st.button("Run"):
    # Convert tickers to a list
    tickers_list = [ticker.strip().upper() for ticker in tickers.split(",")]

    # Initialize an empty DataFrame to store close prices
    price_data = pd.DataFrame()

    # Fetch data for each ticker
    for ticker in tickers_list:
        data = obb.equity.price.historical(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
        )

        # Convert OBBObject to DataFrame and select the 'close' column
        df = data.to_df()
        price_data[ticker] = df["close"]

    if not price_data.empty:
        # Plot the prices using Plotly
        st.subheader("Asset Prices (OpenBB)")
        price_data.index = pd.to_datetime(price_data.index)
        fig = px.line(price_data, x=price_data.index, y=price_data.columns, labels={"x": "Date", "value": "Price", "variable": "Ticker"})
        st.plotly_chart(fig)

        # Plot the returns for simply buying and holding the S&P 500
        st.subheader("S&P 500 Price")
        sp500_data = obb.equity.price.historical(
            symbol="SPY",
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        ).to_df()
        if "close" in sp500_data.columns:
            fig_sp500 = px.line(sp500_data, x=sp500_data.index, y="close", labels={"x": "Date", "close": "Price"})
            st.plotly_chart(fig_sp500)
    else:
        st.error("No data fetched. Please check your ticker symbols and date range.")

    st.subheader("Hierarchical Portfolio")
    st.text("Plot is a test sample for best HRP/HERC model with cross-validation for hyperparamters")

    hrp = HierarchicalRiskParity(
            risk_measure=RiskMeasure.CVAR, 
            hierarchical_clustering_estimator=HierarchicalClustering(),
            portfolio_params=dict(name="HRP-CVaR-Ward-Pearson")
            )

    herc = HierarchicalEqualRiskContribution(
                risk_measure=RiskMeasure.CVAR,
                hierarchical_clustering_estimator=HierarchicalClustering(),
                portfolio_params=dict(name="HERC-CVaR-Ward-Pearson")
                )

    models = [hrp, herc]
    comparator = Comparator(price_data, models)
    comparator.run()

    testing_start_date = comparator.population[0].returns_df.index[0]

    returns_fig = comparator.population.plot_cumulative_returns()

    benchmark_returns = sp500_data["close"].pct_change().dropna()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
    benchmark_returns = benchmark_returns[benchmark_returns.index>=testing_start_date]

    benchmark_trace = go.Scatter(x=benchmark_returns.index, y=benchmark_returns.cumsum(), name="S&P Hold Benchmark")
    
    returns_fig.add_trace(benchmark_trace)
    st.plotly_chart(returns_fig)

    
    best_models = comparator.best_models
    names = [model.portfolio_params["name"] for model in best_models]
    preds = [model.predict(comparator.X_test) for model in best_models]

    cols = st.columns(len(preds))
    for c, p in zip(cols, preds):
        with c:
            _fig = p.plot_composition()
            st.plotly_chart(_fig)
            st.table(p.composition)

    tabs = st.tabs(names)
    for t, p in zip(tabs, preds):
        with t:
            summary = p.summary()
            annual_summary = summary[[i for i in summary.index if any([_i in i for _i in ["Annualized", "CVaR", "EVaR", "CDaR"]])]]
            
            tab_annual, tab_full = st.tabs(["Annualized", "Full"])
            with tab_annual:
                st.table(annual_summary.to_dict())
            
            with tab_full:
                st.table(summary.to_dict())