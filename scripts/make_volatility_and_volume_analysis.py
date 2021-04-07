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
import sys
def assign_percentiles(bars : pd.DataFrame, quantiles = 20, statistic = "range") -> pd.DataFrame:
    bars["quantile"] = pd.qcut(bars[statistic].rank(method='first'), quantiles,
                               labels=[i for i in range(1, quantiles + 1)])
    return bars

def match_bars_and_trades(trades_path, bars_path, active_passive_table_path, heatmap_path,heatmap_csv_path,write_quantiles,n_quantiles = 20):
    full_trades_data = trades.MatchedOrdersData(trades_path)

    for target_quant in [write_quantiles]:
        trades_data = copy.deepcopy(full_trades_data) # We copy the trades data objet so we can modify it

         # Read trades
        trades_data.matched_orders.index = trades_data.matched_orders["DateTime"] # make datetime index

        bars = pd.read_csv(bars_path) # Read Bars

        # Assign Quantiles
        bars = assign_percentiles(bars, n_quantiles)
        bars = bars[bars["quantile"] == target_quant].reset_index()
        windows = list()
        for i in range(len(bars)-1):
            windows.append([bars["first"].loc[i],bars["last"].loc[i]])

        #select trades in windows
        trades_data.select_data_in_windows(windows)
        passive_active_stat = trades_data.passive_active_stat()
        passive_active_stat.to_csv(active_passive_table_path)

        trades_data.strip_agent_numbers()

        nodes,edges = make_traded_volume_matrix(trades_data, cutoff = 0.0)

        directed_graph = nx.DiGraph()  # Rows are the active agent
        directed_graph.add_nodes_from(nodes)
        directed_graph.add_weighted_edges_from(edges)

        # Calculate the total volume traded by each node
        calculate_total_node_volume(directed_graph)
        normalize_directional_graph(directed_graph)


        volume_table = nx.to_pandas_adjacency(directed_graph, weight="volume")
        volume_table.to_csv(heatmap_csv_path)
        fig,ax = plt.subplots()
        ax = sns.heatmap(volume_table.apply(np.log10).replace([np.inf, -np.inf], np.nan).fillna(0), linewidths=.5, ax = ax)
        fig.tight_layout()
        fig.savefig(heatmap_path)

if __name__ == "__main__":
    trades_path = sys.argv[1]
    bars_path = sys.argv[2]
    active_passive_table_path = sys.argv[3]
    heatmap_path = sys.argv[4]
    heatmap_csv_path = sys.argv[5]
    write_quantiles = int(sys.argv[6])
    # data_path: "testing/test_data" compression : none or "gzip"
    match_bars_and_trades(trades_path,
                          bars_path,
                          active_passive_table_path,
                          heatmap_path,
                          heatmap_csv_path,
                          write_quantiles,
                          n_quantiles = 20)