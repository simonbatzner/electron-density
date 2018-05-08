from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from KRR_reproduce import *

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# setup
ev2kcal = 1 / 0.043                     # conversion factor
SIM_NO = 150                            # total number of data points
SEED = 42
M = 140                                 # number of training points

# path to data
os.environ['PROJDIR'] = '/Users/simonbatzner1/Desktop/Research/Research_Code/ML-electron-density'
STR_PREF = os.environ['PROJDIR'] + '/data/H2_DFT/temp_data/store/'

# print results
filename = 'RF_KS_SEP' + '_gridsearch.txt'


ens = []
seps = []
fours = []

for n in range(SIM_NO):

    # load separation, energy, and density
    sep = np.load(STR_PREF + 'sep_store/sep' + str(n) + '.npy')
    en = np.load(STR_PREF + 'en_store/en' + str(n) + '.npy')
    four = np.load(STR_PREF + 'four_store/four' + str(n) + '.npy')

    # format
    sep = np.reshape(sep, (1,))[0]
    en = np.reshape(en, (1,))[0]['energy']
    four = np.real(four)

    ens.append(en)
    seps.append(sep)
    fours.append(four)

data = seps
labels = ens

# define training and test indices
train_indices = [int(n) for n in np.round(np.linspace(0, 149, M))]
test_indices = [n for n in range(150) if n not in train_indices]

if len(train_indices) != M:
    print("Size of training set doesn't match the M specified")

x_train = np.array([data[n] for n in train_indices])
x_test = np.array([data[n] for n in test_indices])
y_train = np.array([labels[n] for n in train_indices])
y_test = np.array([labels[n] for n in test_indices])
x_seps_test = np.array([seps[n] for n in test_indices])

# convert to np arrays
x_train = np.array(x_train)
x_train = x_train.reshape(-1,1)
x_test = np.array(x_test)
x_test = x_test.reshape(-1,1)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(x_train.shape)
print(y_test.shape)

param_grid = {"n_estimators": [10, 50, 100, 200, 500, 1000, 5000], "max_depth": [10, 25, 50]}
#param_grid = {"n_estimators": [10, 20], "max_depth": [10, 20]}

# cross - validation
reg = GridSearchCV(RandomForestRegressor(random_state=SEED), param_grid=param_grid,
                   scoring='neg_mean_squared_error', verbose=10, cv=5)

# train
reg.fit(x_train, y_train)

# eval on training data
y_true_train, y_pred_train = y_train, reg.predict(x_train)

# eval on test data
y_true, y_pred = y_test, reg.predict(x_test)



with open(filename, 'a') as fp:
    fp.write("\nBest parameters set found on development set:")
    fp.write((str(reg.best_params_)))
    fp.write("\n\nMAE on training data in [kcal/mol]: {}".format(mean_absolute_error(y_true_train, y_pred_train) * ev2kcal))
    fp.write("MAE on test data in [kcal/mol]:\t {}".format(mean_absolute_error(y_true, y_pred) * ev2kcal))

