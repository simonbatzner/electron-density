from unittest import TestCase
from Jon_Production.regression import GaussianProcess
import numpy as np
import pprint

class TestGP_PF(TestCase):
    def setUp(self):
        self.gp = GaussianProcess(training_data=np.random.rand(1,10), training_labels=np.random.rand(10,1),
                                  test_data=np.random.rand(1,5), test_labels=np.random.rand(5,1), sigma=10, n_restarts=10,
                                  sklearn=False, verbosity=2)


    def test_train(self):
        self.setUp()
        pprint.pprint(dir(self.gp))
        # self.gp.train()
        # self.fail()

    # def test_opt_hyper(self):
    #     # self.fail()
    #
    # def test_inference(self):
    #     # self.fail()
