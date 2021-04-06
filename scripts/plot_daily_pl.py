import matplotlib.pylab as plt
import pandas as pd
import sys
from helpers import get_column_colors
#import mpld3
#from mpld3 import plugins

def make_figure(input_paths):
    other_agents = [
        "MarketMaker",
        "RT",
        "SectorRotateInst",
        "LongShortInst",
        "PairsTrader"]

    fig, ax = plt.subplots(len(other_agents), figsize=(6, 12))
    agent_pls = list()
    for input_file in input_paths:
        agent_pls.append(pd.read_csv(input_file))
    total_agent_pl = pd.concat(agent_pls).groupby(level=0).sum() # https://stackoverflow.com/questions/38940946/average-of-multiple-dataframes-with-the-same-columns-and-indices
    for agent_name,current_ax in zip(other_agents,ax):
        data = total_agent_pl.filter(regex=agent_name)
        if data.empty == False:
            colors = get_column_colors(data)
            data.reset_index(drop = True).plot(ax= current_ax, legend = False, color = colors)
        current_ax.set_title(agent_name)
        current_ax.grid(True, alpha=0.3)
    return fig,ax

def make_mpl_daily_pl_plot(output_path, input_paths):
    fig,ax = make_figure(input_paths)
    plt.tight_layout()

    fig.savefig(output_path)

if __name__ == "__main__":
    output_path = sys.argv[1]
    input_paths = sys.argv[2:] # data_path: "testing/test_data" compression : none or "gzip"
    make_mpl_daily_pl_plot(output_path, input_paths)

