#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""" GP KS-Mapping

    # References:
        [1] Brockherde et al. Bypassing the Kohn-Sham equations with machine learning. Nature Communications 8, 872 (2017)
        [2] http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
        [3] http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py

Simon Batzner, Steven Torrisi, Jon Vandermause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from matplotlib import pyplot as plt


# suppress sklearn warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, ExpSineSquared
from KRR_reproduce import *

# params
kernel_choice = 'rbf'  # specify either c_rbf, rbf, matern or expsinesquared
m = 5  # number of training points

# setup
ev2kcal = 1 / 0.043  # conversion factor
sim_no = 150  # total number of data points
np.random.seed(1)

# path to data
os.environ['PROJDIR'] = '/Users/simonbatzner1/Desktop/Research/Research_Code/ML-electron-density'
STR_PREF = os.environ['PROJDIR'] + '/data/H2_DFT/temp_data/store/'

# load data
ens = []
seps = []

for n in range(sim_no):
    # load separation, energy, and density
    sep = np.load(STR_PREF + 'sep_store/sep' + str(n) + '.npy')
    en = np.load(STR_PREF + 'en_store/en' + str(n) + '.npy')

    # format
    sep = np.reshape(sep, (1,))[0]
    en = np.reshape(en, (1,))[0]['energy']

    ens.append(en)
    seps.append(sep)

# set up training and test data
data = seps
labels = ens

train_indices = [int(n) for n in np.round(np.linspace(0, 149, m))]
test_indices = [n for n in range(150) if n not in train_indices]

if len(train_indices) != m:
    print("Size of training set doesn't match the m specified")

x_train = np.array([data[n] for n in train_indices])
x_test = np.array([data[n] for n in test_indices])
y_train = np.array([labels[n] for n in train_indices])
y_test = np.array([labels[n] for n in test_indices])

# convert to np arrays
x_train = np.array(x_train)
x_train = x_train.reshape(-1, 1)
x_test = np.array(x_test)
x_test = x_test.reshape(-1, 1)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_test_list = [data[n] for n in test_indices]
x_train_list = [data[n] for n in train_indices]

# build gp w/ a noiseless kernel and print properties
kernel_mat = [C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
              RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2)),
              Matern(length_scale=10, length_scale_bounds=(1e-2, 1e2),
                     nu=10),
              ExpSineSquared(length_scale=1.0, periodicity=3.0,
                             length_scale_bounds=(1e-2, 1e2),
                             periodicity_bounds=(1e-2, 1e2))]

for fig_index, kernel in enumerate(kernel_mat):

    #build gp
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)

    #plot prior
    plt.figure(fig_index, figsize=(8, 8))
    plt.subplot(2, 1, 1)
    X_ = np.array(seps)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')
    y_samples = gp.sample_y(X_[:, np.newaxis], 10)
    plt.plot(X_, y_samples, lw=1)
    plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

    # fit
    gp.fit(x_train, y_train)

    # inference
    x_train_list = [data[n] for n in train_indices]
    y_pred_test = gp.predict(np.atleast_2d(x_train_list).T)

    # plot posterior
    plt.subplot(2, 1, 2)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')

    y_samples = gp.sample_y(X_[:, np.newaxis], 10)
    plt.plot(X_, y_samples, lw=1)

    plt.scatter(x_train[:, 0], y_pred_test, c='r', s=10, zorder=10, edgecolors=(0, 0, 0))
    plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
              % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),
              fontsize=12)
    plt.tight_layout()

plt.show()

