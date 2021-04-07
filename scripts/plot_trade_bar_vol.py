import matplotlib.pylab as plt
import pandas as pd
import sys
from make_volatility_analysis import assign_percentiles
from helpers import get_column_colors
#import mpld3
#from mpld3 import plugins

def make_figure(input_path):
    print(input_path)
    bars = pd.read_csv(input_path)

    # Assign Quantiles
    bars = assign_percentiles(bars, 20)
    fig, ax = plt.subplots(2)
    for pct in [1, 20]:
        ax[0].plot(bars[bars["quantile"] == pct]["simTime"].values, bars[bars["quantile"] == pct]["range"].values, ".",
                   label="ptc = {}, n = {}".format(pct, len(bars[bars["quantile"] == pct])))
    ax[0].legend()
    ax[0].set_title("Range Quantiles")
    ax[0].set_xlabel("sim time")

    bars = assign_percentiles(bars, 20, statistic="return")
    for pct in [1, 20]:
        ax[1].plot(bars[bars["quantile"] == pct]["simTime"].values, bars[bars["quantile"] == pct]["return"].values, ".",
                   label="ptc = {}, n = {}".format(pct, len(bars[bars["quantile"] == pct])))
    ax[1].legend()
    ax[1].set_title("Log Returns Quantiles")
    ax[1].set_xlabel("sim time")

    return fig,ax

def make_mpl_trade_range_plot(output_path, input_paths):
    fig,ax = make_figure(input_paths)
    plt.tight_layout()

    fig.savefig(output_path)

if __name__ == "__main__":
    output_path = sys.argv[1]
    input_paths = sys.argv[2] # data_path: "testing/test_data" compression : none or "gzip"
    make_mpl_trade_range_plot(output_path,input_paths)

