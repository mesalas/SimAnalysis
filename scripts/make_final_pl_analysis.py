import sys
#from helpers.data_conf import make_data_conf

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#from helpers.get_exchange_and_sim import get_exchange_and_sim

def get_fast_to_slow_pl_ratio(input_paths):
    agent_pls = list()
    for input_file in input_paths:
        agent_pls.append(pd.read_csv(input_file).iloc[-1])
    total_agent_pl = pd.concat(agent_pls).groupby(
        level=0).sum()  # https://stackoverflow.com/questions/38940946/average-of-multiple-dataframes-with-the-same-columns-and-indices
    mean_mm_pl = total_agent_pl[total_agent_pl.index.str.contains("MarketMaker-")].mean()
    mean_mm_fast_pl = total_agent_pl[total_agent_pl.index.str.contains("MarketMakerFast-")].mean()
    return mean_mm_fast_pl,mean_mm_pl


def make_final_pl_analysis(dispersed_inputs,intermediate_inputs, concentrated_input_paths, output_path):
    #fig, ax = plt.subplots()
    concentrated_fast_pl = list()
    concentrated_slow_pl = list()
    for input in concentrated_input_paths:
        fast, slow = get_fast_to_slow_pl_ratio(input)
        concentrated_fast_pl.append(fast)
        concentrated_slow_pl.append(slow)

    intermediate_fast_pl = list()
    intermediate_slow_pl = list()
    for input in intermediate_inputs:
        fast, slow = get_fast_to_slow_pl_ratio(input)
        intermediate_fast_pl.append(fast)
        intermediate_slow_pl.append(slow)

    dispersed_fast_pl = list()
    dispersed_slow_pl = list()
    for input in dispersed_inputs:
        fast, slow = get_fast_to_slow_pl_ratio(input)
        dispersed_fast_pl.append(fast)
        dispersed_slow_pl.append(slow)

    labels = ["dispersed", "intermediate", "concentrated"]
    fast = [0,np.mean(intermediate_fast_pl),np.mean(concentrated_fast_pl)]
    slow = [np.mean(dispersed_slow_pl), np.mean(intermediate_slow_pl),np.mean(concentrated_slow_pl)]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, fast, width, label='Fast')
    rects2 = ax.bar(x + width / 2, slow, width, label='Slow')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('PL')
    #ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

#    ax.bar_label(rects1, padding=3)
#    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    fig.savefig(output_path)




# if __name__ == "__main__":
#     input_path, freq, output_path = sys.argv[1:] # data_path: "testing/test_data" compression : none or "gzip"
#     if input_path.split(".")[-1] == "gz":
#         compression = "gzip"
#     else:
#         compression = None
#
#     make_midprice(input_path, freq, output_path)


