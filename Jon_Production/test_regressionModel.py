from unittest import TestCase
from oo_util import RegressionModel, GaussianProcess
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np


class TestRegressionModel(TestCase):

    def setUp(self):

            training_data = np.random.rand(100, 4)
            test_data = np.random.rand(20, 4)
            training_labels = np.random.rand(100, 1)
            test_labels = np.random.rand(20, 1)

            self.model = GaussianProcessRegressor()

            self.GP = RegressionModel(model=self.model, training_data=training_data, training_labels=training_labels,
                                      test_data=test_data, test_labels=test_labels, model_type='gp', verbosity=2)

    def test_train(self):
        self.GP.train()
        assert(self.GP.verbosity == 2)

    def test_inference(self):
        print(1)
        print(self.GP.inference().shape)
        assert(self.GP.inference().shape == (20,))

