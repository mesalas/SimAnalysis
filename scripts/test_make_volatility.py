import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
from make_volatility_analysis import make_volatility_analysis,match_bars_and_trades

class MyTestCase(unittest.TestCase):
    def test_make_volatility_analysis(self):
        make_volatility_analysis("testing/test_data/reduced_data/0_ABC_bars_15T.csv.gz")
        match_bars_and_trades()

if __name__ == '__main__':
    unittest.main()
