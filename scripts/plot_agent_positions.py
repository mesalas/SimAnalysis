import matplotlib.pylab as plt
import pandas as pd
import sys
import mpld3
from mpld3 import plugins

def make_agent_positions_figure(input_files):

    fig, ax = plt.subplots()
    for input_file in input_files:
        agent = input_file.split('_')[-1].split(".")[0]
        try:
            agent_pos = pd.read_csv(input_file, compression = "gzip")
        except:
            print(input_file)
            #continue
        #color, lw = _get_agent_plot_style(data_conf["agent to color map"], agent)
        ax.plot(agent_pos["simTime"],agent_pos["0"],label = agent)# color=color, label=agent, linewidth=lw)
        #color, lw = _get_agent_plot_style(data_conf["agent to color map"],agent)
    return fig,ax

def make_mpl_position_plot(output_path, input_paths):
    fig,ax = make_agent_positions_figure(input_paths)
    ax.legend()

    fig.savefig(output_path)

def make_mpld3_position_plot(output_path, input_paths):
    fig,ax = make_agent_positions_figure(input_paths)
    handles, labels = ax.get_legend_handles_labels() # return lines and labels
    interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,
                                                             ax.lines),
                                                         labels,
                                                         alpha_unsel=.1,
                                                         alpha_over=1.5,
                                                         start_visible=True)
    plugins.connect(fig, interactive_legend)

    html = mpld3.fig_to_html(fig)
    with open(output_path, "w") as html_file:
        html_file.write(html)