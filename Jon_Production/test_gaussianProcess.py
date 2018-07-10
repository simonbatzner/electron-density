from unittest import TestCase
from oo_util import GaussianProcess
import numpy as np


class TestGaussianProcess(TestCase):

    def setUp(self):
        training_data = np.random.rand(100, 4)
        test_data = np.random.rand(20, 4)
        training_labels = np.random.rand(100, 1)
        test_labels = np.random.rand(20, 1)

        self.GP = GaussianProcess(training_data, training_labels, test_data, test_labels, kernel='rbf', length_scale=10,
                                  length_scale_min=1e-2, length_scale_max=1e2, n_restarts=10, sklearn=True, verbosity=2)

    def test_train(self):
        self.setUp()
        self.GP.train()

    def test_inference(self):
        self.setUp()
        print(self.GP.inference().shape)
        assert(self.GP.inference().shape == (20,))
