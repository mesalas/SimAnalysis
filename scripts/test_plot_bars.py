import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
from plot_trade_bar_vol import make_mpl_trade_range_plot

class MyTestCase(unittest.TestCase):

    def test_plot_trade_bars_vol(self):

        input = "testing/test_data/reduced_data/0_ABC_bars_1T.csv.gz"
        output = "testing/test_data/figures/0_ABC_volatility_1T.png"

        make_mpl_trade_range_plot(output, input)

if __name__ == '__main__':
    unittest.main()
