import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
from make_volatility_and_volume_analysis import match_bars_and_trades

class MyTestCase(unittest.TestCase):
    def test_make_volatility_analysis(self):
        #make_volatility_analysis("testing/test_data/reduced_data/0_ABC_bars_15T.csv.gz")
        trades_path = "testing/test_data/ABC_NYSE@0_Matching-MatchedOrders.csv"
        bars_path = "testing/test_data/reduced_data/0_ABC_bars_1T.csv.gz"

        match_bars_and_trades(trades_path,bars_path,
                              "testing/test_data/reduced_data/q1_active_passive_table.csv",
                              "testing/test_data/figures/q1_heatmap.png",
                              "testing/test_data/reduced_data/q1_heatmap.csv",
                              1)
        match_bars_and_trades(trades_path,bars_path,
                              "testing/test_data/reduced_data/q20_active_passive_table.csv",
                              "testing/test_data/figures/q20_heatmap.png",
                              "testing/test_data/reduced_data/q20_heatmap.csv",
                              20)

if __name__ == '__main__':
    unittest.main()
