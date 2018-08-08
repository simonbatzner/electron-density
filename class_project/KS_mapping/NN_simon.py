#!/usr/bin/env python

"""" ML mapping from external potential to charge density - AP275 Class Project, Harvard University

    # References:
        [1] Brockherde et al. Bypassing the Kohn-Sham equations with machine learning. Nature Communications 8, 872 (2017)

Simon Batzner, Steven Torrisi, Jon Vandermause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import argparse

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from math import sqrt
from KRR_reproduce import *
# from generate_H2_data import *


def load_data():
    """
    Load separations, energies, densities
    """
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
    print(input_shape)

    # hidden layers
    model.add(Dense(hidden_size[0], input_shape=input_shape, activation=activation))
    for layer_size in hidden_size[1:]:
        model.add(Dense(layer_size, activation=activation))
        model.add(Dropout(0.2))

    # output layer
    model.add(Dense(1, activation='linear'))

    if summary:
        model.summary()

    return model


def train(model, training_data, training_labels, validation_data, validation_labels, batchsize=64):
    """"
    Train Neural Network model
    """
    history = model.fit(training_data, training_labels, validation_data=(validation_data, validation_labels),
                        batch_size=batchsize,
                        verbose=0, shuffle=True)
    return history


def main():
    seed = 42

    # params
    hidden = (10,)
    epochs = 500000
    test_size = 0.95
    learning_rate = 0.01
    decay = 0.0

    ens, seps, fours = load_data()

    # create list of gaussian potentials
    print("Building potentials...")
    pots = []
    grid_len = 5.29177 * 2

    for n in range(SIM_NO):
        dist = seps[n]
        pot = pot_rep(dist, grid_len, grid_space=0.8)
        pot = pot.flatten()
        pots.append(pot)

    # setup training and test data
    data = pots
    labels = ens
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=seed)

    # keras input needs numpy ndarrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # neural net
    model = init_architecture(input_shape=pots[0].shape, hidden_size=tuple(hidden), summary=True,
                              activation='tanh')

    adam = optimizers.Adam(lr=learning_rate, decay=decay)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mse'])
    # model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mse'])

    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=500,
                       verbose=0, mode='auto')

    tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=1, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    history = model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_split=0.2, callbacks=[es, tb])

    # predict
    test_loss = model.evaluate(x_test, y_test)

    # eval on training data
    y_true, y_pred = y_train, model.predict(x_train)
    print("\nMSE on training data: \t{}".format(mean_squared_error(y_true, y_pred)))

    # eval on test data
    y_true, y_pred = y_test, model.predict(x_test)
    print("MSE on test data: \t\t{}".format(mean_squared_error(y_true, y_pred)))

    # Predict on new data
    print("\n\t\tPred \t| \tTrue\n")
    print(np.c_[y_pred, y_true])

    # results
    print("\n\nTest Loss: {}".format(test_loss[1]))

    print(history.history.keys())
    # summarize history for loss
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    global SIM_NO, STR_PREF, TEST

    # ignore tf warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    SIM_NO = 150

    # path to data
    os.environ['PROJDIR'] = '/Users/simonbatzner1/Desktop/Research/Research_Code/ML-electron-density'
    STR_PREF = os.environ['PROJDIR'] + '/data/H2_DFT/temp_data/store/'
    TEST = np.load(os.environ['PROJDIR'] + '/data/H2_DFT/temp_data/store/sep_store/sep149.npy')

    main()
