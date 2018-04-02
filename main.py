#!/usr/bin/env python

"""" ML mapping from external potential to charge density - AP275 Class Project, Harvard University

    # Arguments

        input_dir: directory containing subdirectories of files
        type: neural network of kernel ridge regression model, specify: 'nn' or 'krr'
        output_dir: directory to write figures and output data to
        hidden: list or tuple containing neural network architecture, e.g. [30, 30] creates a neural network w/ 2 layers a 30 neurons
        nfolds: int, number of folds for k-fold cross validation
        train_min: int, first subdirectory to read files from
        train_max: int, last subdirectory to read files from
        test_size: float in [0, 1], ratio of test to overall data
        summary: bool, yes or no to display summary of neural network architecture

    # References:
        [1] Brockherde et al. Bypassing the Kohn-Sham equations with machine learning. Nature Communications 8, 872 (2017)
        [2] KRR work based on http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-plot-kernel-ridge-regression-py (Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>)

Simon Batzner, Steven Torrisi, Jon Vandermause

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from ase.io import read
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from math import sqrt


class System(object):
    def __init__(self, filename, struc):
        self.filename = filename
        self.struc = struc
        self.label = self.struc.get_potential_energy()
        self.real_charge = self.read_charge()
        self.k_charge = self.to_recip()

    def read_charge(self):
        """
        read charge density from QE charge density file
        """
        pass

    def to_recip(self):
        """
        convert real space charge density to k-space charge density
        """
        self.k_charge = np.fft.fftn(self.real_charge)
        self.k_charge = self.k_charge / self.struc.get_volume()

    def info(self):
        print(self.filename)


def load_data(input_dir, train_min, train_max):
    """
    Read in data from file
    :param input_dir: directroy to read training and test data from
    :param train_min: starting subdirectory to read from
    :param train_max: final subdirectory to read from
    :return: list of systems
    """
    systems = []

    def get_fnames(dir, train_min, train_max):
        train_list = [str(i) for i in list(range(train_min, train_max + 1))]
        r = []
        for root, dirs, files in os.walk(dir):
            for dir in dirs:
                if dir in train_list:
                    r.append(os.path.join(root, dir, ''))  # add QE filename suffix
        return r

    fnames = get_fnames(dir=input_dir, train_min=train_min, train_max=train_max)

    for filename in fnames:
        struc = read(filename=filename, format='espresso-out')
        System(filename=filename, struc=struc)
        systems.append(System)

    return systems


def setup_data(systems):
    """
    Split data into input and corresponding labels
    :param systems: list of structures to train/ test on
    :return: input data X and labels
    """
    data = []
    labels = []
    for system in systems:
        data.append(system.real_charge)
        labels.append(system.label)

    return data, labels


def init_architecture(input_shape, hidden_size, summary, activation='relu'):
    """
    Built Neural Network using Keras

    :param input_shape: shape of the input data
    :param hidden_size: tuple of number of hidden layers, eg. (30, 30, 40) builds a network with hidden layers 30-30-40
    :param summary: boolean, true plots a summary
    :param activation: activiation function
    :return: keras Sequential model
    """
    model = Sequential()
    model.add(Dense(hidden_size[0], input_dim=input_shape, activation=activation))
    for layer_size in hidden_size[1:]:
        model.add(Dense(layer_size, activation=activation))
        model.add(Dense(1, activation='sigmoid'))

    if summary:
        model.summary()

    return model


def set_loss(model, loss, optimizer, metrics):
    """"
    Set loss function, optimizer (add metric for classification tasks)
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train(model, training_data, training_labels, validation_data, validation_labels, batchsize=64):
    """"
    Train Neural Network model
    """
    history = model.fit(training_data, training_labels, validation_data=(validation_data, validation_labels),
                        batch_size=batchsize,
                        verbose=1, shuffle=True)
    return history


def inference(model, type, x_test, y_test):
    """"
    Compute test loss RMSE and predicted targets using trained model
    """

    # SIMON: -- TEST EQUIVALENCY OF KERAS RMSE METRIC AND SELF-IMPLEMENTED
    if type == 'nn':
        loss = sqrt(model.evaluate(x_test, y_test, verbose=2)[1])
        y_predict_test = model.predict(x_test, verbose=2)

    elif type == 'krr':
        y_predict_test = model.predict(x_test)
        loss = sqrt(mean_squared_error(y_test, y_predict_test))

    return loss, y_predict_test


def main():
    seed = 42

    # arguments
    parser = argparse.ArgumentParser(
        description='Machine Learning the ground-state charge density')
    parser.add_argument('--input_dir', type=str, default='./')
    parser.add_argument('--type', type=str, default='')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--hidden', nargs='+', type=int)
    parser.add_argument('--nfolds', type=int, default=5)
    parser.add_argument('--train_min', type=int, default=1)
    parser.add_argument('--train_max', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--summary', type=bool, default=False)
    args = parser.parse_args()

    if not ((args.type.lower() == 'nn' or args.type.lower() == 'krr')):
        raise ValueError("model type must be either 'nn' or 'krr'")

    if not 0 <= args.test_size <= 1:
        raise ValueError('test size must be in [0, 1]')

    # load DFT output data from file into list of Systems objects
    systems = load_data(args.input_dir, train_min=args.train_min, train_max=args.train_max)

    # define input and labels
    data, labels = setup_data(systems)

    x_trainval, x_test, y_trainval, y_test = train_test_split(data, labels, test_size=args.test_size, random_state=seed)

    # split trainval into training and validation data
    # k-fold cross-validation, each fold is used once as a validation while the k - 1 remaining are used for training
    kf = KFold(n_splits=args.nfolds, shuffle=True, random_state=seed)
    history = []

    # NEURAL NETWORK
    if args.type.lower() == 'nn':
        model = init_architecture(input_dim=data[1].shape[0], hidden_size=tuple(args.hidden), summary=args.summary,
                                  activation=args.activation)
        set_loss(model=model, loss='mean_squared_error', optimizer='Adam', metrics=['mse'])

        # train NN
        for train_index, val_index in kf.split(x_trainval):
            history.append(
                train(model=model, training_data=x_trainval[train_index], training_labels=y_trainval[train_index],
                      validation_data=x_trainval[val_index],
                      validation_labels=y_trainval[val_index]))


    # KERNEL RIDGE REGRESSION
    elif args.type.lower() == 'krr':

        model = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=args.nfolds,
                             param_grid={"alpha": [1, 0.1, 0.01, 0.01],
                                         "gamma": np.logspace(-2, 2, 10)})

        model.fit(x_trainval, y_trainval)

    # predict
    loss, y_predict_test = inference(model=model, type=args.type.lower(), x_test=x_test, y_test=y_test)

    # results
    print("Test loss: {}".format(loss))

    # plot
    plt.plot(x_test, y_test)
    plt.plot(x_test, y_predict_test)
    plt.xlabel('input')
    plt.ylabel('target')
    plt.title('True vs Prediction on test data')
    plt.savefig('Test_Pred')


if __name__ == "__main__":
    main()
