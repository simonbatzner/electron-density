from unittest import TestCase
from oo_util import KernelRidgeRegression
import numpy as np

class TestKernelRidgeRegression(TestCase):


    def setUp(self):
        training_data = np.random.rand(100, 4)
        test_data = np.random.rand(20, 4)
        training_labels = np.random.rand(100, 1)
        test_labels = np.random.rand(20, 1)

        self.krr = KernelRidgeRegression(training_data, training_labels, test_data, test_labels, kernel='rbf',
                                         alpha_range=[1e0, 0.1, 1e-2, 1e-3], gamma_range=np.logspace(-2, 2, 5),
                                         cv=5, sklearn=True, verbosity=2)


    def test_train(self):
        self.setUp()
        self.krr.train()

    def test_inference(self):
        self.setUp()
        self.krr.train()
        assert(self.krr.inference().shape == (20,) or self.krr.inference().shape == (20, 1))
