import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
import make_midprices as make_midprices
import plot_midprice as plot_midprice

from helpers.data_conf import make_data_conf

class MyTestCase(unittest.TestCase):
    def test_resample_and_plot_data(self):
        inputs = ["testing/test_data/ABC_NYSE@0_Matching-OrderBook.csv",
                  "testing/test_data/DEF_NYSE@0_Matching-OrderBook.csv",
                  "testing/test_data/GHI_NYSE@0_Matching-OrderBook.csv"]
        outputs = ["testing/test_data/reduced_data/0_ABC_mp_5T.csv.gz",
                   "testing/test_data/reduced_data/0_DEF_mp_5T.csv.gz",
                   "testing/test_data/reduced_data/0_GHI_mp_5T.csv.gz"]

        [make_midprices.make_midprice(input_file, "5T", output_file) for input_file,output_file in zip(inputs,outputs)]

        inputs = outputs
        output = "testing/test_data/figures/0_mp_5T.png"

        plot_midprice.make_mpl_midprice_plot(output, inputs)

        output = "testing/test_data/figures/0_mp_5T.html"
        plot_midprice.make_mpld3_midprice_plot(output, inputs)


if __name__ == '__main__':
    unittest.main()
