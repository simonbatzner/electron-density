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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, ExpSineSquared
from KRR_reproduce import *
from Solid_State.KRR_Functions import *

# params
kernel_choice = 'rbf'  # specify either c_rbf, rbf, matern or expsinesquared
m = 7  # number of training points
input_dim = 12

# setup
ev2kcal = 1 / 0.043  # conversion factor
sim_no = 201
np.random.seed(1)

# path to data
os.environ['PROJDIR'] = '/Users/simonbatzner1/Desktop/Research/Research_Code/ML-electron-density'
STR_PREF = os.environ['PROJDIR'] + '/Aluminium_Dataset/Store/'

# load data
pos = []
ens = []
fours = []

for n in range(sim_no):
    # load arrays
    en_curr = np.reshape(np.load(STR_PREF + 'en_store/energy' + str(n) + '.npy'), (1))[0]
    pos_curr = np.load(STR_PREF + 'pos_store/pos' + str(n) + '.npy')
    four_curr = np.load(STR_PREF + 'four_store/four' + str(n) + '.npy')

    # store arrays
    ens.append(en_curr)
    pos_curr = pos_curr.flatten()
    pos.append(pos_curr)
    fours.append(four_curr)

# convert to np arrays
ens = np.array(ens)
pos = np.array(pos)
fours = np.abs(np.array(fours))


[x_train, x_test, y_train, y_test, train_fours, test_fours] = get_train_test(m, sim_no, input_dim, \
                                                                                     pos, ens, fours)


# build gp w/ a noiseless kernel and print properties
kernel_dict = {
    'c_rbf': C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
    'rbf': RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2)),
    'matern': Matern(length_scale=10, length_scale_bounds=(1e-2, 1e2),
                     nu=10),
    'expsinesquared': ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                     length_scale_bounds=(1e-2, 1e2),
                                     periodicity_bounds=(1e-2, 1e2))}

kernel = kernel_dict[kernel_choice]

print("\nKernel: {}".format(kernel))
print("Hyperparameters: \n")

for hyperparameter in kernel.hyperparameters: print(hyperparameter)
print("Parameters:\n")

params = kernel.get_params()
for key in sorted(params): print("%s : %s" % (key, params[key]))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)

# fit
print("\nFitting GP...")
gp.fit(x_train, y_train)

# inference
y_pred_test = gp.predict(x_test)
y_pred_train = gp.predict(x_train)
y_pred_all, sigma = gp.predict(pos, return_std=True)

# print results
print("\n=============================================")
print("MAE on training data in [kcal/mol]: \t{}".format(mean_absolute_error(y_train, y_pred_train) * ev2kcal))
print("MAE on test data in [kcal/mol]: \t\t{}".format(mean_absolute_error(y_test, y_pred_test) * ev2kcal))

# plot w/ confidence intervals
fig = plt.figure()
plt.plot(x_train, y_train, 'r.', markersize=10, label=u'Training Data')
plt.plot(x_test, y_pred_test, 'b-', label=u'Predictions')
plt.fill(np.concatenate([pos, pos[::-1]]),
         np.concatenate([y_pred_all - 1.9600 * sigma,
                         (y_pred_all + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='upper left')

plt.show()
