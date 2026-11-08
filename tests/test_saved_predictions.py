"""Regression tests for the four quantitative thesis results."""
import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).resolve().parents[1]/'scripts/evaluate_saved_predictions.py'
s=importlib.util.spec_from_file_location('evaluate_saved_predictions',p)
evalmod=importlib.util.module_from_spec(s)
s.loader.exec_module(evalmod)

class SavedPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref,cls.pred=evalmod.load_arrays()
        cls.errors=evalmod.calculate_relative_errors(cls.ref,cls.pred)

    def test_grid(self):
        self.assertEqual(self.pred.shape,(51,500,4))

    def test_relative_l2_metrics(self):
        expected={'alpha':0.2363,'X':0.1505,'rho0':10.9334,'v':0.9997}
        for k,value in expected.items():
            with self.subTest(k=k):
                self.assertAlmostEqual(self.errors[k],value,delta=0.00005)

if __name__=='__main__':
    unittest.main()
