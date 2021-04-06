import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import trades
from make_network import make_traded_volume_matrix,make_directed_graph,save_directed_graph

def assign_percentiles(bars : pd.DataFrame, quantiles = 5, statistic = "range") -> pd.DataFrame:
    bars["quantile"] = pd.qcut(bars[statistic].rank(method='first'), quantiles,
                               labels=[i for i in range(1, quantiles + 1)])
    return bars

def plot_vol_and_range(bars):

    fig,ax = plt.subplots(2)
    ax[0].hist(bars["range"], bins=25)
    ax[1].hist(np.log(bars["close"])-np.log(bars["open"]), bins=25)
    fig.savefig("testing/test_data/figures/0_ABC_bars_vol.png")

def make_volatility_analysis(input_path):
    bars = pd.read_csv(input_path)

    # Assign Quantiles
    bars = assign_percentiles(bars, 5)
    fig,ax = plt.subplots(2)
    for pct in [1, 2, 3, 4, 5]:
        ax[0].plot(bars[bars["quantile"] == pct]["simTime"].values,bars[bars["quantile"] == pct]["range"].values,".", label = "ptc = {}, n = {}".format(pct,len(bars[bars["quantile"] == pct])))
    ax[0].legend()
    ax[0].set_title("Range Quantiles")
    ax[0].set_xlabel("sim time")
    ax[0].set_ylabel("15 min range")

    bars = assign_percentiles(bars, 5, statistic="return")
    for pct in [1, 2, 3, 4, 5]:
        ax[1].plot(bars[bars["quantile"] == pct]["simTime"].values,bars[bars["quantile"] == pct]["return"].values,".", label = "ptc = {}, n = {}".format(pct,len(bars[bars["quantile"] == pct])))
    ax[1].legend()
    ax[1].set_title("Log Returns Quantiles")
    ax[1].set_xlabel("sim time")
    ax[1].set_ylabel("15 min range")
    fig.savefig("testing/test_data/figures/test_pct.png")

    bars = assign_percentiles(bars, 5)
    fig,ax = plt.subplots()
    for pct in [1, 2, 3, 4, 5]:
        ax.plot(bars[bars["quantile"] == pct]["simTime"].values,bars[bars["quantile"] == pct]["open"].values,".", label = "ptc = {}, n = {}".format(pct,len(bars[bars["quantile"] == pct])))
    ax.legend()
    ax.set_title("Range Quantiles")
    ax.set_xlabel("sim time")
    ax.set_ylabel("15 min range")
    fig.savefig("testing/test_data/figures/open_quantiles.png")

def match_bars_and_trades():
    trades_path = "testing/test_data/ABC_NYSE@0_Matching-MatchedOrders.csv"
    bars_path = "testing/test_data/reduced_data/0_ABC_bars_15T.csv.gz"
    trades_data = trades.MatchedOrdersData(trades_path)
    trades_data.matched_orders.index = trades_data.matched_orders["DateTime"] # make datetime index
    bars = pd.read_csv(bars_path)

    # Assign Quantiles
    bars = assign_percentiles(bars, 5)
    bars = bars[bars["quantile"] == 5].reset_index()
    windows = list()
    for i in range(len(bars)-1):
        windows.append([bars["first"].loc[i],bars["last"].loc[i]])

    trades_data.select_data_in_windows(windows)
    traded_volume_matrix = make_traded_volume_matrix(trades_data, cutoff = 0.0)
    directed_graph = make_directed_graph(traded_volume_matrix[0],traded_volume_matrix[1])
    save_directed_graph(directed_graph, "testing/test_data/reduced_data/5_quant_network.gefx")
    print("x")



