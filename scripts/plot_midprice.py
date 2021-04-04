import matplotlib.pylab as plt
import pandas as pd
import sys
#import mpld3
#from mpld3 import plugins

def make_figure(input_paths):
    fig,ax = plt.subplots()
    for input_file in input_paths:
        symbol = input_file.split("/")[-1].split("_")[1]
        mid_price = pd.read_csv(input_file)
        ax.plot(mid_price["simTime"], mid_price["midprice"], label = symbol)
    ax.grid(True, alpha=0.3)
    return fig,ax

def make_mpl_midprice_plot(output_path, input_paths):
    fig,ax = make_figure(input_paths)
    ax.legend()

    fig.savefig(output_path)

# def make_mpld3_midprice_plot(output_path, input_paths):
#     fig,ax = make_figure(input_paths)
#     handles, labels = ax.get_legend_handles_labels() # return lines and labels
#     interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,
#                                                              ax.lines),
#                                                          labels,
#                                                          alpha_unsel=.1,
#                                                          alpha_over=1.5,
#                                                          start_visible=True)
#     plugins.connect(fig, interactive_legend)
#
#     html = mpld3.fig_to_html(fig)
#     with open(output_path, "w") as html_file:
#         html_file.write(html)

if __name__ == "__main__":
    output_path = sys.argv[1]
    input_paths = sys.argv[2:] # data_path: "testing/test_data" compression : none or "gzip"
    make_mpl_midprice_plot(output_path, input_paths)

