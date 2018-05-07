#!/usr/bin/env python

"""" Hyperparameter optimization for the Random Forest KS - Mapping using Gaussian Potentials

    AP275 Class Project, Harvard University
    # References:
        [1] Brockherde et al. Bypassing the Kohn-Sham equations with machine learning. Nature Communications 8, 872 (2017)

Simon Batzner, Steven Torrisi, Jon Vandermause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from KRR_reproduce import *


def load_data():
    print("Loading data...")
    ens = []
    seps = []
    fours = []

    for n in range(SIM_NO):
        # load separation, energy, and density
        sep = np.load(STR_PREF + 'sep_store/sep' + str(n) + '.npy')
        en = np.load(STR_PREF + 'en_store/en' + str(n) + '.npy')
        four = np.load(STR_PREF + 'four_store/four' + str(n) + '.npy')

        # put results in a nicer format
        sep = np.reshape(sep, (1,))[0]
        en = np.reshape(en, (1,))[0]['energy']
        four = np.real(four)

        # store quantities
        ens.append(en)
        seps.append(sep)
        fours.append(four)
    return ens, seps, fours


def main():
    seed = 42

    # params
    test_size = 0.1
    ens, seps, fours = load_data()
    grid_space_min = 0.5
    grid_space_max = 1.0
    grid_space_list = np.linspace(grid_space_min, grid_space_max, 6)
    print(grid_space_list)

    for grid_space in grid_space_list:

        print("Grid space: {}".format(grid_space))
        # create list of gaussian potentials
        print("Building potentials...")
        pots = []
        grid_len = 5.29177 * 2

        for n in range(SIM_NO):
            dist = seps[n]
            pot = pot_rep(dist, grid_len, grid_space=grid_space)
            pot = pot.flatten()
            pots.append(pot)

        # setup training and test data
        data = pots
        labels = ens
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=seed)

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        reg = GridSearchCV(RandomForestRegressor(),
                           param_grid={"n_estimators": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                                       "max_depth": [10, 20, 30, 40, 50, 100, 200]},
                           scoring='neg_mean_absolute_error', verbose=10, cv=5)

        # train
        reg.fit(x_train, y_train)

        # eval on training data
        y_true_train, y_pred_train = y_train, reg.predict(x_train)

        # eval on test data
        y_true, y_pred = y_test, reg.predict(x_test)

        with open('RF_gridsearch.txt', 'a') as fp:
            fp.write("\n=======================\n")
            fp.write("Best parameters set found for grid_space = {} on development set:\n".format(grid_space))
            fp.write(json.dumps(reg.best_params_))
            fp.write("\n\nMAE on training data: {}\n".format(mean_absolute_error(y_true_train, y_pred_train)))
            fp.write("\tMAE on test data: {}\n".format(mean_absolute_error(y_true, y_pred)))


if __name__ == "__main__":
    global SIM_NO, STR_PREF, TEST

    SIM_NO = 150

    # path to data
    os.environ['PROJDIR'] = '/home/sbatzner'
    STR_PREF = os.environ['PROJDIR'] + '/store/'

    main()
