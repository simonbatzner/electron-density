#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""" Gaussian Process Regression model

Implementation is based on Algorithm 2.1 (pg. 19) of
"Gaussian Processes for Machine Learning" by Rasmussen and Williams

Simon Batzner
"""
import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


class GaussianProcess:
    """
    Gaussian Process Regression Model
    """

    def __init__(self, kernel):
        """
        Initialize GP parameters and training data

        :param: kernel  (func) func specifying used GP kernel
        """

        # predictive mean and variance
        self.pred_mean = None
        self.pred_var = None

        # gp kernel and hyperparameters
        self.kernel = kernel
        self.length_scale = None
        self.sigma_n = None
        self.sigma_f = None

        # quantities used in GPR algorithm
        self.k_mat = None
        self.l_mat = None
        self.alpha = None

        # training set
        self.training_data = []
        self.training_labels = []

    def train(self):
        """
        Train Gaussian Process model on training data
        """

        # optimize hyperparameters
        self.opt_hyper()

        # following: Algorithm 2.1 (pg. 19) of
        # "Gaussian Processes for Machine Learning" by Rasmussen and Williams
        self.set_kernel(sigma_f=self.sigma_f, length_scale=self.length_scale, sigma_n=self.sigma_n)

        # get alpha and likelihood
        self.set_alpha()

    def opt_hyper(self):
        """
        Optimize hyperparameters of GP by minimizing minus log likelihood
        """
        # initial guess
        x_0 = np.array([self.sigma_f, self.length_scale, self.sigma_n])

        # nelder-mead optimization
        res = minimize(self.minus_like_hyp, x_0, method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': True})

        self.sigma_f, self.length_scale, self.sigma_n = res.x[0], res.x[1], res.x[2]

    def predict(self, xt, d):
        """ Make GP prediction with SE kernel """

        # get kernel vector
        kv = self.get_kernel_vector(x=xt, d_1=1)

        # get predictive mean
        self.pred_mean = np.matmul(kv.transpose(), self.alpha)

        # get predictive variance
        v = solve_triangular(self.l_mat, kv, lower=True)
        self_kern = self.kernel(xt, xt, d, d, self.sigma_f, self.length_scale)
        self.pred_var = self_kern - np.matmul(v.transpose(), v)

    def minus_like_hyp(self, hyp):
        """
        Get minus likelihood as a function of hyperparameters

        :param hyp          list of hyperparameters to optimize
        :return minus_like  negative likelihood
        """
        like = self.like_hyp(hyp)
        minus_like = -like

        return minus_like

    def like_hyp(self, hyp):
        """
        Get likelihood as a function of hyperparameters

        :param  hyp      hyperparameters to be optimized
        :return like    likelihood
        """

        # unpack hyperparameters
        sigma_f = hyp[0]
        length_scale = hyp[1]
        sigma_n = hyp[2]

        # calculate likelihood
        self.set_kernel(sigma_f=sigma_f, length_scale=length_scale, sigma_n=sigma_n)
        self.set_alpha()
        like = self.get_likelihood()

        return like

    def get_likelihood(self):
        """
        Get log marginal likelihood

        :return like    likelihood
        """
        like = -(1 / 2) * np.matmul(self.training_labels.transpose(), self.alpha) - \
               np.sum(np.log(np.diagonal(self.l_mat))) - np.log(2 * np.pi) * self.k_mat.shape[1] / 2

        return like

    def set_kernel(self, sigma_f, length_scale, sigma_n):
        """
        Compute 3Nx3N noiseless kernel matrix
        """
        d_s = ['xrel', 'yrel', 'zrel']

        # initialize matrix
        size = len(self.training_data) * 3
        k_mat = np.zeros([size, size])

        # calculate elements
        for m_index in range(size):
            x_1 = self.training_data[int(math.floor(m_index / 3))]
            d_1 = d_s[m_index % 3]
            for n_index in range(m_index, size):
                x_2 = self.training_data[int(math.floor(n_index / 3))]
                d_2 = d_s[n_index % 3]

                # calculate kernel
                cov = self.kernel(x_1, x_2, d_1, d_2, sigma_f, length_scale)
                k_mat[m_index, n_index] = cov
                k_mat[n_index, m_index] = cov

        # perform cholesky decomposition and store
        self.l_mat = np.linalg.cholesky(k_mat + sigma_n ** 2 * np.eye(size))
        self.k_mat = k_mat

    def get_kernel_vector(self, x, d_1):
        """
        Compute kernel vector
        """
        ds = ['xrel', 'yrel', 'zrel']
        size = len(self.training_data) * 3
        kv = np.zeros([size, 1])

        for m in range(size):
            x2 = self.training_data[int(math.floor(m / 3))]
            d_2 = ds[m % 3]
            kv[m] = self.kernel(x, x2, d_1, d_2, self.sigma_f, self.length_scale)

        return kv

    def set_alpha(self):
        """
        Set weight vector alpha
        """
        ts1 = solve_triangular(self.l_mat, self.training_labels, lower=True)
        alpha = solve_triangular(self.l_mat.transpose(), ts1)

        self.alpha = alpha
