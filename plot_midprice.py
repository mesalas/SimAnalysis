import matplotlib.pylab as plt
import pandas as pd
import sys
def make_mpl_midprice_plot(output_path, input_paths):
    fig,ax = plt.subplots()
    for input_file in input_paths:
        mid_price = pd.read_csv(input_file)
        ax.plot(mid_price["simTime"], mid_price["midprice"])
    fig.savefig(output_path)

if __name__ == "__main__":
    output_path, input_paths = sys.argv[1:] # data_path: "testing/test_data" compression : none or "gzip"
    make_mpl_midprice_plot(output_path, input_paths)