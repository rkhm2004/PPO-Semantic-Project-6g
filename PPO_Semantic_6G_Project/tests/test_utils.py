import unittest
import os
from src.utils import plot_results, log_results

class TestUtils(unittest.TestCase):
    def test_plot_results(self):
        plot_results([1.0], [0.5], "test_plot.png")
        self.assertTrue(os.path.exists("test_plot.png"))
        os.remove("test_plot.png")

    def test_log_results(self):
        log_results(1.0, 0.5, "test_log.txt")
        self.assertTrue(os.path.exists("test_log.txt"))
        with open("test_log.txt", "r") as f:
            self.assertIn("Avg URLLC Latency", f.read())
        os.remove("test_log.txt")

if __name__ == '__main__':
    unittest.main()