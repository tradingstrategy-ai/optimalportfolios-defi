"""Default example from OptimaPortfolios README"""
import os
import pickle
from dataclasses import dataclass, astuple
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple
import qis as qis

from optimalportfolios import compute_rolling_optimal_weights, PortfolioObjective, Constraints

#: Where to store the results figures and such
OUTPUT_PATH = Path("./output")

DATASET_PATH = Path("~/exported/ethereum-1d.parquet")

@dataclass
class PortfolioRunInput:
    #: daily prices, assets as columns
    prices: pd.DataFrame
    #: daily prices, assets as columns
    benchmark: pd.DataFrame
    #: Map assets to their groups.
    #: Asset symbol -> group name mapping.
    group_data: pd.Series


def convert_to_asset_price_series(
    df: pd.DataFrame,
    asset_whitelist: set[str] | None = None,
    benchmark_assets: set[str] | None = None,
) -> PortfolioRunInput:

    # Sample data:
    #                             ticker  timestamp        open        high         low       close        volume           tvl  base quote  fee                                               link  pair_id  buy_tax  sell_tax
    # 0       WETH-USDC-uniswap-v2-30bps 2020-05-05  205.587587  205.587587  201.486251  201.486251  1.100000e-02  9.890000e-01  WETH  USDC   30  https://tradingstrategy.ai/trading-view/ethere...        1      0.0       0.0
    # 1       WETH-USDC-uniswap-v2-30bps 2020-05-06  201.078391  201.358458  201.078391  201.358458  1.689000e-03  9.886890e-01  WETH  USDC   30  https://tradingstrategy.ai/trading-view/ethere...        1      0.0       0.0

    # Filter by trading pair base token
    if asset_whitelist is not None:
        df = df.loc[df["base"].isin(asset_whitelist)]

    # If we have multiple trading pairs for the same asset drop others except the lowest fee tier
    # Group by base and quote, and find the index of the row with the minimum fee for each pair
    # First, group by base and quote, then find the minimum fee for each pair
    min_fee_by_pair = df.groupby(['base', 'quote'])['fee'].min().reset_index()
    # Then merge this back with the original dataframe to identify rows with the lowest fee
    lowest_fee_df = pd.merge(df, min_fee_by_pair, on=['base', 'quote', 'fee'])

    close_prices = lowest_fee_df[["base", "close", "timestamp"]]

    #  aggfunc='mean' -> Handle duplicates
    close_prices = close_prices.pivot_table(index='timestamp', columns='base', values='close', aggfunc='mean')

    benchmark_prices = close_prices[list(benchmark_assets)]

    # Add holding cash option
    close_prices["USD"] = 1 + np.random.uniform(
        -1e-15,
        1e-15,
        size=len(close_prices),
    )

    # TODO: Add CoinGecko grouping later
    group_data = {}
    for name in close_prices.columns:
        group_data[name] = "DeFi"
    group_data = pd.Series(group_data)

    return PortfolioRunInput(
        prices=close_prices,
        benchmark=benchmark_prices,
        group_data=group_data,
    )


def main():

    print("Setting up prices")
    df = pd.read_parquet(DATASET_PATH)
    portfolio_run_input = convert_to_asset_price_series(
        df,
        asset_whitelist={"WBTC", "WETH", "AAVE"},
        benchmark_assets={"WBTC"},
    )
    prices, benchmark_prices, group_data = astuple(portfolio_run_input)

    # 2. get universe data
    time_period = qis.TimePeriod(
        prices.index[0],
        prices.index[-1]
    )   # period for computing weights backtest

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


if __name__ == "__main__":
    main()