import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import rl_train

class TestRLTrain(unittest.TestCase):
    def test_run_rl(self):
        # Should return a string (path) or None if dependencies are missing
        result = rl_train.run_rl()
        self.assertTrue(isinstance(result, str) or result is None)

if __name__ == "__main__":
    unittest.main()
