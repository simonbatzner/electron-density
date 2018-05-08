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
ev2kcal = 1/0.043     # conversion factor
SEED = 42
SIM_NO = 201          # total number of data points
M = 180               # number of training points

# print results
filename = 'RF_KS_SEP_Al1' + '_gridsearch.txt'


# path to data
os.environ['PROJDIR'] = '/Users/simonbatzner1/Desktop/Research/Research_Code/ML-electron-density'
STR_PREF = os.environ['PROJDIR']+'/Aluminium_Dataset/Store/'

# load MD results
MD_pos = np.load(STR_PREF+'MD_positions.npy')
MD_ens = np.load(STR_PREF+'MD_energies.npy')

ens = []
seps = []
fours = []

pos = []
ens = []
fours = []

for n in range(SIM_NO):
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
fours = np.array(fours)

data = pos
labels = ens

# define training and test indices
train_indices = [int(n) for n in np.round(np.linspace(0, 200, M))]
test_indices = [n for n in range(201) if n not in train_indices]

if len(train_indices) != M:
    print("Size of training set doesn't match the M specified")

x_train = np.array([data[n] for n in train_indices])
x_test = np.array([data[n] for n in test_indices])
y_train = np.array([labels[n] for n in train_indices])
y_test = np.array([labels[n] for n in test_indices])

# convert to np arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_test.shape)


#param_grid = {"n_estimators": [10, 50, 100, 200, 500, 1000, 5000], "max_depth": [10, 25, 50]}
param_grid = {"n_estimators": [10, 20], "max_depth": [10, 20]}

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
    fp.write("\n\nMAE on training data in [eV]: {}".format(mean_absolute_error(y_true_train, y_pred_train)))
    fp.write("MAE on test data in [eV]:\t {}".format(mean_absolute_error(y_true, y_pred)))

