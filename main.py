#!/usr/bin/env python

"""" ML mapping from external potential to charge density - AP275 Class Project, Harvard University

References:
    [1] Brockherde et al. Bypassing the Kohn-Sham equations with machine learning. Nature Communications 8, 872 (2017)

Simon Batzner, Steven Torrisi, Jon Vandermause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
from keras.models import Sequential
from keras.layers import (Activation, Dense, LeakyReLU)
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from ase.build import *
from ase.io import read, write


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



def load_data(input_dir):
    """
    Read data from file
    :param input_dir: directroy to read training and test data from
    :return: list of systems
    """
    rootdir = input_dir
    systems = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == '':  # add QE filename suffix
                struc = read(filename=os.path.join(subdir, file), format='espresso-out')
                System(filename=os.path.join(subdir, file), struc=struc)
                systems.append(System)

    return systems


def setup_data(systems):
    """
    Split data into input and corresponding labels
    :param systems: list of structures to train/ test on
    :return: input data X and labels
    """
    X = []
    labels = []
    for system in systems:
        X.append(system.real_charge)
        labels.append(system.label)

    return X, labels


def init_architecture(X, hidden_size, summary, mode='regression', activation='relu'):
    """
    Setup model layout
    :param X: input data
    :param hidden_size: tuple of number of hidden layers, eg. (30, 30, 40) builds a network with hidden layers 30-30-40
    :param summary: boolean, true plots a summary
    :param mode: regression or classification
    :param activation: activiation function
    :return: keras Sequential model
    """
    model = Sequential()
    model.add(Dense(hidden_size[0], input_dim=X.shape[1], activation=activation))
    for layer_size in hidden_size[1:]:
        model.add(Dense(layer_size, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
    return model


def set_loss(model, loss, optimizer):
    """"
    Set loss function, optimizer (add metric for classification tasks)
    """
    model.compile(optimizer=optimizer, loss=loss)
    return model


def train(model, training_data, training_labels, validation_data, validation_labels, epochs, batchsize=64):
    """"
    Train model
    """
    history = model.fit(training_data, training_labels, validation_data=(validation_data, validation_labels),
                        batch_size=batchsize,
                        epochs=epochs, verbose=1, shuffle=True)
    return history


def inference(model, X_test, Y_test):
    """"
    Compute and print test accuracy using trained model
    """
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)


def main():
    seed = 42

    # arguments
    parser = argparse.ArgumentParser(
        description='Neural Network approach to learning the ground-state charge density')
    parser.add_argument('--input_dir', type=str, default='./')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--hidden', nargs='+', type=int)
    parser.add_argument('--mode', type=str, default='regression')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--nfolds', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--hidden', nargs='+', type=int)
    parser.add_argument('--summary', type=bool, default=False)
    args = parser.parse_args()

    # load
    systems = load_data(args.input_dir)

    # define input and labels
    X, labels = setup_data(systems)

    # build model
    model = init_architecture(X=data, hidden_size=args.hidden, summary=args.summary, mode=args.mode,
                              activation=args.activation)
    set_loss(model=model, loss='mean_squared_error', optimizer='Adam')

    # visualize
    plot_model(model, show_shapes=True,
               to_file=os.path.join(args.output_dir, 'model.png'))

    # train/val vs. test split
    x_trainval, x_test, y_trainval, y_test = train_test_split(X, labels, test_size=args.test_size, random_state=seed)

    # train vs. val split
    # k-fold cross-validation, each fold is used once as a validation while the k - 1 remaining are used for training
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    history = []

    # train
    for train_index, val_index in kf.split(x_trainval):
        history.append(
            train(model=model, training_data=x_trainval[train_index], training_labels=y_trainval[train_index],
                  validation_data=x_trainval[val_index],
                  validation_labels=y_trainval[val_index]), epochs=args.epochs)

    # predict
    inference(model=model, X_test=x_test, Y_test=y_test)


if __name__ == "__main__":
    main()
