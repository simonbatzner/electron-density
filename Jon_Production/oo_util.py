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


class RegressionModel:
    """Base class for regression models"""
    def __init__(self, model, training_data, training_labels, test_data, test_labels, model_type):
        """
        Initialization
        """
        self.model = model
        self.training_data = training_data
        self.training_labels = training_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.model_type = type

    def train(self):
        """
        Train ML model on training_data/ training_labels
        """
        self.model.fit(self.training_data, self.training_labels)

    def inference(self):
        """
        Predict on test data
        """
        self.model.predict(self.test_data)


class GaussianProcess(RegressionModel):
    """Gaussian Process Regression Model"""
    def __init__(self, training_data, training_labels, test_data, test_labels, kernel, length_scale, length_scale_min,
                 length_scale_max, n_restarts, sklearn):
        """
        Initialization
        """
        self.length_scale = length_scale
        self.length_scale_min = length_scale_min
        self.length_scale_max = length_scale_max
        self.n_restarts = n_restarts
        self.sklearn = sklearn
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
                                 training_labels=training_labels, test_labels=test_labels, model_type='gp')

    def train(self):
        """
        Train ML model on training_data/ training_labels
        """
        super(GaussianProcess, self).train()

    def inference(self):
        """
        Predict on test data
        """
        super(GaussianProcess, self).inference()


class KernelRidgeRegression(RegressionModel):
    """KRR Regression Model"""
    def __init__(self, training_data, training_labels, test_data, test_labels, sklearn, kernel,
                 alpha_range, gamma_range):
        """
        Initialization
        """
        self.alpha_range = alpha_range
        self.gamma_range = gamma_range
        self.kernel = kernel
        self.sklearn = sklearn

        if self.sklearn:
            self.model = KernelRidge(kernel=kernel)

        else:
            # PyFly implementation of Kernel Ridge Regression
            self.model = KRR_PF

        RegressionModel.__init__(self, model=self.model, training_data=training_data, test_data=test_data,
                                 training_labels=training_labels, test_labels=test_labels, model_type='krr')

    def train(self, alpha_range, gamma_range, cv):
        """
        Train using gridsearch on KRR
        """
        kr = GridSearchCV(KernelRidge(kernel=self.kernel), cv=cv,
                          param_grid={"alpha": alpha_range,
                                      "gamma": gamma_range})

        kr.fit(self.training_data, self.training_labels)

    def inference(self):
        """
        Predict on test data
        """
        super(KernelRidgeRegression, self).inference()


class VGP():
    """Vectorized Gaussian Process"""
    pass


class KRR_PF():
    """PyFly implementation of Kernel Ridge Regression"""
    pass