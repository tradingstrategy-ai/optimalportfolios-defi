"""Default example from OptimaPortfolios README"""
import os
import pickle
from pathlib import Path

import pandas as pd
import yfinance as yf
from typing import Tuple
import qis as qis

from optimalportfolios import compute_rolling_optimal_weights, PortfolioObjective, Constraints

#: Cache YFinance price data for subsequent runs here
PRICES_CACHE_PATH = Path("/tmp/prices.pickle")

#: Where to store the results figures and such
OUTPUT_PATH = Path("./output")


def fetch_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    fetch universe data for the portfolio construction:
    1. dividend and split adjusted end of day prices: price data may start / end at different dates
    2. benchmark prices which is used for portfolio reporting and benchmarking
    3. universe group data for portfolio reporting and risk attribution for large universes
    this function is using yfinance to fetch the price data
    """
    universe_data = dict(SPY='Equities',
                         QQQ='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         IEF='Bonds',
                         LQD='Credit',
                         HYG='HighYield',
                         GLD='Gold')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)
    if PRICES_CACHE_PATH.exists():
        print("Using cached prcies")
        prices = pickle.load(PRICES_CACHE_PATH.open("rb"))
    else:
        print("Starting prices download from YFinance")
        prices = yf.download(tickers, start=None, end=None, ignore_tz=True)['Close']
        pickle.dump(prices, PRICES_CACHE_PATH.open("wb"))

    prices = prices[tickers]  # arrange as given
    prices = prices.asfreq('B', method='ffill')  # refill at B frequency
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, group_data

# 2. get universe data
prices, benchmark_prices, group_data = fetch_universe_data()
time_period = qis.TimePeriod('31Dec2004', '16Aug2024')   # period for computing weights backtest

# 3.a. define optimisation setup
print("Define optimisation")
portfolio_objective = PortfolioObjective.MAX_DIVERSIFICATION  # define portfolio objective
returns_freq = 'W-WED'  # use weekly returns
rebalancing_freq = 'QE'  # weights rebalancing frequency: rebalancing is quarterly on WED
span = 52  # span of number of returns_freq-returns for covariance estimation = 12y
constraints0 = Constraints(is_long_only=True,
                           min_weights=pd.Series(0.0, index=prices.columns),
                           max_weights=pd.Series(0.5, index=prices.columns))

# 3.b. compute solvers portfolio weights rebalanced every quarter
weights = compute_rolling_optimal_weights(prices=prices,
                                          portfolio_objective=portfolio_objective,
                                          constraints0=constraints0,
                                          time_period=time_period,
                                          rebalancing_freq=rebalancing_freq,
                                          span=span)

# 4. given portfolio weights, construct the performance of the portfolio
print("Assigning weights")
funding_rate = None  # on positive / negative cash balances
rebalancing_costs = 0.0010  # rebalancing costs per volume = 10bp
weight_implementation_lag = 1  # portfolio is implemented next day after weights are computed
portfolio_data = qis.backtest_model_portfolio(prices=prices.loc[weights.index[0]:, :],
                                              weights=weights,
                                              ticker='MaxDiversification',
                                              funding_rate=funding_rate,
                                              weight_implementation_lag=weight_implementation_lag,
                                              rebalancing_costs=rebalancing_costs)

# 5. using portfolio_data run the reporting with strategy factsheet
# for group-based reporting set_group_data
print("Generating reports")
portfolio_data.set_group_data(group_data=group_data, group_order=list(group_data.unique()))
# set time period for portfolio reporting
figs = qis.generate_strategy_factsheet(portfolio_data=portfolio_data,
                                       benchmark_prices=benchmark_prices,
                                       time_period=time_period,
                                       **qis.fetch_default_report_kwargs(time_period=time_period))


if not OUTPUT_PATH.exists():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

# save report to pdf and png
qis.save_figs_to_pdf(figs=figs,
                     file_name=f"{portfolio_data.nav.name}_portfolio_factsheet",
                     orientation='landscape',
                     local_path=str(OUTPUT_PATH))

qis.save_fig(fig=figs[0], file_name=f"example_portfolio_factsheet1", local_path=str(OUTPUT_PATH))
if len(figs) > 1:
    qis.save_fig(fig=figs[1], file_name=f"example_portfolio_factsheet2", local_path=str(OUTPUT_PATH))

