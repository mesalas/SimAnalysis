import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import trades
from make_network import make_traded_volume_matrix,make_directed_graph,save_directed_graph
from make_micro_network_analysis import calculate_total_node_volume,normalize_directional_graph
import networkx as nx
import seaborn as sns
from nxviz.plots import CircosPlot
import copy

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


        #c = CircosPlot(directed_graph,edge_width="weight",node_labels=True, node_size=0.0)
        #c.draw()
        #plt.savefig("{}_test.png".format(target_quant))
        #save_directed_graph(directed_graph, "testing/test_data/reduced_data/{}_quant_network.gexf".format(target_quant))


#    print("x")



