import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
import make_average_networks as make_average_networks
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



#from helpers.data_conf import make_data_conf
def make_plot(directed_graph_with_cliques,ax):
    volumes = [directed_graph_with_cliques.nodes[node]["volume"] for node in directed_graph_with_cliques.nodes]
    community = [directed_graph_with_cliques.nodes[node]["community"] for node in directed_graph_with_cliques.nodes]
    labels = {node:node for node in directed_graph_with_cliques.nodes}
    widths = [directed_graph_with_cliques.edges[edge]["volume"] for edge in directed_graph_with_cliques.edges]

    #label offsets
    pos_higher = {}
    y_off = 1  # offset on the y axis
    pos = nx.circular_layout(directed_graph_with_cliques)
    x = list()
    y = list()
    for k, v in pos.items():
        x.append(v[0]*1.5*1.3)
        y.append(v[1]*1.1*1.2)
        pos_higher[k] = (v[0]*1.5, v[1]*1.1)

    nx.draw(directed_graph_with_cliques,pos, node_size = 500.*np.array(volumes)/np.max(volumes),
                     width= 0.2+10.*np.array(widths)/np.max(widths), alpha = 0.75, ax=ax, node_color = community, cmap=plt.cm.Accent)
    nx.draw_networkx_labels(directed_graph_with_cliques,pos_higher,labels, ax=ax)
    ax.set_ylim([min(y), max(y)])
    ax.set_xlim([min(x), max(x)])


class MyTestCase(unittest.TestCase):
    def test_pl_analysis(self):
        fig,ax = plt.subplots(nrows=2, ncols=3, figsize = (20,10))
        dispersed_inputs = ["../temp_data/run93/batch1/working/reduced_data/1_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch2/working/reduced_data/2_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch3/working/reduced_data/3_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch4/working/reduced_data/4_ABC_1q_volume_heatmap_1T.csv"]

        intermediate_inputs = ["../temp_data/run93/batch5/working/reduced_data/5_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch6/working/reduced_data/6_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch7/working/reduced_data/7_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch8/working/reduced_data/8_ABC_1q_volume_heatmap_1T.csv"]

        concentrated_inputs = ["../temp_data/run93/batch9/working/reduced_data/9_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch10/working/reduced_data/10_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch11/working/reduced_data/11_ABC_1q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch12/working/reduced_data/12_ABC_1q_volume_heatmap_1T.csv"
                  ]


        graph_output = "../temp_data/run93/dispersed_ABC_1q_1T.gexf"
        summary_output = "../temp_data/run93/dispersed_ABC_1q_1T.csv"

        directed_graph_with_cliques = make_average_networks.make_average_networks(graph_output,summary_output,dispersed_inputs)
        make_plot(directed_graph_with_cliques, ax[0][0])
        graph_output = "../temp_data/run93/concentrated_ABC_1q_1T.gexf"
        summary_output = "../temp_data/run93/concentrated_ABC_1q_1T.csv"

        directed_graph_with_cliques =make_average_networks.make_average_networks(graph_output,summary_output,concentrated_inputs)
        make_plot(directed_graph_with_cliques, ax[0][1])
        graph_output = "../temp_data/run93/intermediate_ABC_1q_1T.gexf"
        summary_output = "../temp_data/run93/intermediate_ABC_1q_1T.csv"

        directed_graph_with_cliques =make_average_networks.make_average_networks(graph_output,summary_output,intermediate_inputs)
        make_plot(directed_graph_with_cliques, ax[0][2])
        dispersed_inputs = ["../temp_data/run93/batch1/working/reduced_data/1_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch2/working/reduced_data/2_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch3/working/reduced_data/3_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch4/working/reduced_data/4_ABC_20q_volume_heatmap_1T.csv"]

        intermediate_inputs = ["../temp_data/run93/batch5/working/reduced_data/5_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch6/working/reduced_data/6_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch7/working/reduced_data/7_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch8/working/reduced_data/8_ABC_20q_volume_heatmap_1T.csv"]

        concentrated_inputs = ["../temp_data/run93/batch9/working/reduced_data/9_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch10/working/reduced_data/10_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch11/working/reduced_data/11_ABC_20q_volume_heatmap_1T.csv",
                  "../temp_data/run93/batch12/working/reduced_data/12_ABC_20q_volume_heatmap_1T.csv"
                  ]


        graph_output = "../temp_data/run93/dispersed_ABC_20q_1T.gexf"
        summary_output = "../temp_data/run93/dispersed_ABC_20q_1T.csv"

        directed_graph_with_cliques =make_average_networks.make_average_networks(graph_output,summary_output,dispersed_inputs)
        make_plot(directed_graph_with_cliques, ax[1][0])
        graph_output = "../temp_data/run93/concentrated_ABC_20q_1T.gexf"
        summary_output = "../temp_data/run93/concentrated_ABC_20q_1T.csv"

        directed_graph_with_cliques =make_average_networks.make_average_networks(graph_output,summary_output,concentrated_inputs)
        make_plot(directed_graph_with_cliques, ax[1][1])
        graph_output = "../temp_data/run93/intermediate_ABC_20q_1T.gexf"
        summary_output = "../temp_data/run93/intermediate_ABC_20q_1T.csv"

        directed_graph_with_cliques =make_average_networks.make_average_networks(graph_output,summary_output,intermediate_inputs)
        make_plot(directed_graph_with_cliques, ax[1][2])
        fig.tight_layout()
        fig.savefig("../temp_data/run93/networks.png")
if __name__ == '__main__':
    unittest.main()