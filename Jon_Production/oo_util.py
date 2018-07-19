#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name

"""" Regression models

Simon Batzner
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, Matern

from Jon_Production.utility import get_SE_K, GP_SE_alpha, GP_SE_like


class RegressionModel:
    """Base class for regression models"""

    def __init__(self, model, training_data, training_labels, test_data, test_labels, model_type, verbosity):
        """
        Initialization
        """
        self.model = model
        self.training_data = training_data
        self.training_labels = training_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.model_type = type
        self.verbosity = verbosity

    def train(self):
        """
        Train ML model on training_data/ training_labels
        """
        if self.verbosity > 1:
            print("Training model...")
        self.model.fit(self.training_data, self.training_labels)

    def inference(self):
        """
        Predict on test data
        """
        if self.verbosity > 1:
            print("Performing inference")
        return self.model.predict(self.test_data)


class GaussianProcess(RegressionModel):
    """Gaussian Process Regression Model"""

    def __init__(self, training_data, training_labels, test_data, test_labels, kernel, length_scale, length_scale_min,
                 length_scale_max, n_restarts, sklearn, verbosity):
        """
        Initialization
        """
        self.length_scale = length_scale
        self.length_scale_min = length_scale_min
        self.length_scale_max = length_scale_max
        self.n_restarts = n_restarts
        self.sklearn = sklearn
        self.verbosity = verbosity
        self.kernel_dict = {'rbf': RBF(length_scale=self.length_scale,
                                       length_scale_bounds=(self.length_scale_min, self.length_scale_max)),
                            'matern_15': Matern(length_scale=self.length_scale,
                                                length_scale_bounds=(self.length_scale_min, self.length_scale_max),
                                                nu=1.5),
                            'matern_25': Matern(length_scale=self.length_scale,
                                                length_scale_bounds=(self.length_scale_min, self.length_scale_max),
                                                nu=2.5)}
        self.kernel = self.kernel_dict[kernel]

        if self.sklearn:
            self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_restarts)

        else:
            # PyFly implementation of a Vectorial Gaussian Process
            self.model = VGP()

        RegressionModel.__init__(self, model=self.model, training_data=training_data, test_data=test_data,
                                 training_labels=training_labels, test_labels=test_labels, model_type='gp',
                                 verbosity=verbosity)

    def train(self):
        """
        Train ML model on training_data/ training_labels
        """
        super(GaussianProcess, self).train()

    def inference(self):
        """
        Predict on test data
        """
        return super(GaussianProcess, self).inference()


class KernelRidgeRegression(RegressionModel):
    """KRR Regression Model"""

    def __init__(self, training_data, training_labels, test_data, test_labels, kernel,
                 alpha_range, gamma_range, cv, sklearn, verbosity):
        """
        Initialization
        """
        self.alpha_range = alpha_range
        self.gamma_range = gamma_range
        self.kernel = kernel
        self.verbosity = verbosity
        self.sklearn = sklearn
        self.cv = cv

        if self.sklearn:
            self.model = GridSearchCV(KernelRidge(kernel=self.kernel), cv=self.cv,
                                      param_grid={"alpha": self.alpha_range,
                                                  "gamma": self.gamma_range})

        else:
            # PyFly implementation of Kernel Ridge Regression
            self.model = KRR_PF

        RegressionModel.__init__(self, model=self.model, training_data=training_data, test_data=test_data,
                                 training_labels=training_labels, test_labels=test_labels, model_type='krr',
                                 verbosity=verbosity)

    def train(self):
        """
        Train ML model on training_data/ training_labels
        """
        super(KernelRidgeRegression, self).train()

    def inference(self):
        """
        Predict on test data
        """
        return super(KernelRidgeRegression, self).inference()


class VGP():
    """
    PyFly implementation of a vectorial Gaussian Process
    """
    pass


class KRR_PF():
    """PyFly implementation of Kernel Ridge Regression"""
    pass
