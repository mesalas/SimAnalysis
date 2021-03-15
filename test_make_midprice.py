import unittest
import make_midprices
from helpers.data_conf import make_data_conf

class MyTestCase(unittest.TestCase):
    def test_something(self):
        data_conf = make_data_conf(0, {"path" : "", "sim_no" : 0, "compression" : None}, "testing/test_data")
        make_midprices.make_midprice(data_conf, "5T")


if __name__ == '__main__':
    unittest.main()
