#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name, too-many-arguments

"""" Regression models

Simon Batzner
"""
import os
import copy

import numpy as np
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, Matern

from Jon_Production.utility import get_SE_K, GP_SE_alpha, minus_like_hyp, GP_SE_pred, symmetrize_forces
from util.project_pwscf import parse_qe_pwscf_output


def get_files(root_dir):
    """
    Find all files matching *.out

    :param root_dir:    str, dir to walk from
    :return:            list, files in root_dir ending with .out
    """
    matching = []

    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.out'):
                matching.append(os.path.join(root, filename))

    return matching


def parse_output(filename, target):
    """
    Parse QE output file and return new datapoint as input_data, target
    :param filename:            str, QE file to read from
    :return: [data, target]     input config and target for ML model from QE output file
    """
    result = parse_qe_pwscf_output(filename)

    if target == 'f':
        positions, forces = result['positions'], result['forces']

    elif target == 'e':
        raise ValueError("Not implemented yet. Stay tuned.")

    else:
        raise ValueError("No proper ML target defined.")

    return positions, forces


class RegressionModel:
    """Base class for regression models"""

    def __init__(self, model, training_data, test_data, correction_folder,
                 training_folder, model_type, target, force_conv, thresh_perc, verbosity):
        """
        Initialization
        """
        self.model = model
        self.target = target
        self.training_data = training_data
        self.test_data = test_data
        self.model_type = model_type
        self.verbosity = verbosity
        self.training_folder = training_folder
        self.correction_folder = correction_folder
        self.force_conv = force_conv
        self.thresh_perc = thresh_perc
        self.aug_files = []
        self.err_thresh = None

    def upd_database(self, cutoff, eta_lower, eta_upper, eta_length, brav_mat, brav_inv, vec1, vec2, vec3):
        """
        Add new training data from augmentation folder
        """
        for file in get_files(self.correction_folder):

            if file not in self.aug_files:

                self.aug_files.append(file)

                positions, forces = parse_output(file, target=self.target)

                for pos, f in zip(positions, forces):
                    self.aug_and_norm(pos=pos, forces=f, cutoff=cutoff, eta_lower=eta_lower, eta_upper=eta_upper,
                                      eta_length=eta_length, brav_mat=brav_mat, brav_inv=brav_inv, vec1=vec1,
                                      vec2=vec2, vec3=vec3)

    def init_database(self, cutoff, eta_lower, eta_upper, eta_length, brav_mat, brav_inv, vec1, vec2, vec3):
        """
        Init training database from directory
        """
        for file in get_files(self.training_folder):

            positions, forces = parse_output(file, target=self.target)

            for pos, f in zip(positions, forces):
                self.aug_and_norm(pos=pos, forces=f, cutoff=cutoff, eta_lower=eta_lower, eta_upper=eta_upper,
                                  eta_length=eta_length, brav_mat=brav_mat, brav_inv=brav_inv, vec1=vec1,
                                  vec2=vec2, vec3=vec3)

        self.set_error_threshold()

    def set_error_threshold(self):
        """
        Set threshold for predictive variance
        :return:
        """
        # TODO: this should be set in first MD frame, then only compared against
        if self.err_thresh is None:
            self.err_thresh = self.thresh_perc * np.mean(
                np.abs(np.array(self.training_data['forces']) * self.force_conv))
        else:
            return

    def aug_and_norm(self, pos, forces, cutoff, eta_lower, eta_upper, eta_length,
                     brav_mat, brav_inv, vec1, vec2, vec3):
        """
        Augment and normalize
        """

        # augment
        self.augment_database(pos, forces, cutoff, eta_lower, eta_upper, eta_length,
                              brav_mat, brav_inv, vec1, vec2, vec3)

        # normalize forces and symmetry vectors
        self.normalize_force()
        self.normalize_symm()

    def augment_database(self, pos, forces, cutoff, eta_lower, eta_upper, eta_length,
                         brav_mat, brav_inv, vec1, vec2, vec3):
        """
        For a given supercell, calculate symmetry vectors for each atom
        """
        for n in range(len(pos)):
            # get symmetry vectors
            symm_x, symm_y, symm_z = symmetrize_forces(pos, n, cutoff, eta_lower, eta_upper,
                                                       eta_length, brav_mat, brav_inv, vec1, vec2, vec3)

            # append symmetry vectors
            self.training_data['symms'].append(symm_x)
            self.training_data['symms'].append(symm_y)
            self.training_data['symms'].append(symm_z)

            # append force components
            self.training_data['forces'].append(forces[n][0])
            self.training_data['forces'].append(forces[n][1])
            self.training_data['forces'].append(forces[n][2])

    def normalize_symm(self):
        """
        Normalize the symmetry vectors in the training set
        """
        symm_len = len(self.training_data['symms'][0])
        td_size = len(self.training_data['symms'])

        # initialize normalized symmetry vector
        self.training_data['symm_norm'] = copy.deepcopy(self.training_data['symms'])

        # store normalization factors
        self.training_data['symm_facs'] = []

        for m in range(symm_len):
            # calculate standard deviation of current symmetry element
            vec = np.array([self.training_data['symms'][n][m] for n in range(td_size)])
            vec_std = np.std(vec)

            # store standard deviation
            self.training_data['symm_facs'].append(vec_std)

            # normalize the current element
            for n in range(td_size):
                self.training_data['symm_norm'][n][m] = self.training_data['symm_norm'][n][m] / vec_std

    def normalize_force(self):
        """
        Normalize forces
        """
        td_size = len(self.training_data['forces'])

        # initialize normalized force vector
        self.training_data['forces_norm'] = copy.deepcopy(self.training_data['forces'])

        # calculate standard deviation of force components
        vec_std = np.std(self.training_data['forces'])

        # store standard deviation
        self.training_data['force_fac'] = vec_std

        # normalize the forces
        for n in range(td_size):
            self.training_data['forces_norm'][n] = self.training_data['forces_norm'][n] / vec_std


class GaussianProcess(RegressionModel):
    """Gaussian Process Regression Model"""

    def __init__(self, training_data=None, test_data=None, kernel='rbf',
                 length_scale=1, length_scale_min=1e-5, length_scale_max=1e5, sigma=1, n_restarts=10,
                 correction_folder='.', training_folder='.', target='f', force_conv=25.71104309541616,
                 thresh_perc=.2, verbosity=1, sklearn=False):
        """
        Initialization
        """
        self.length_scale = length_scale
        self.sklearn = sklearn
        self.verbosity = verbosity

        # predictions
        self.pred = None
        self.pred_var = None

        if self.sklearn:

            self.length_scale_min = length_scale_min
            self.length_scale_max = length_scale_max
            self.n_restarts = n_restarts

            self.kernel_dict = {'rbf': RBF(length_scale=self.length_scale,
                                           length_scale_bounds=(self.length_scale_min, self.length_scale_max)),
                                'matern_15': Matern(length_scale=self.length_scale,
                                                    length_scale_bounds=(self.length_scale_min, self.length_scale_max),
                                                    nu=1.5),
                                'matern_25': Matern(length_scale=self.length_scale,
                                                    length_scale_bounds=(self.length_scale_min, self.length_scale_max),
                                                    nu=2.5)}
            self.kernel = self.kernel_dict[kernel]
            self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_restarts)

        else:

            # PyFly implementation of a Gaussian Process
            self.model = None
            self.sigma = sigma
            self.K = None
            self.L = None
            self.alpha = None

        RegressionModel.__init__(self, model=self.model, training_data=training_data, test_data=test_data,
                                 correction_folder=correction_folder, training_folder=training_folder,
                                 model_type='gp', target=target, force_conv=force_conv, thresh_perc=thresh_perc,
                                 verbosity=verbosity)

    def retrain(self, cutoff, eta_lower, eta_upper, eta_length, brav_mat, brav_inv, vec1, vec2, vec3):
        """
        Retrain GP model in active learning procedure based on new training data
        """
        self.upd_database(cutoff, eta_lower, eta_upper, eta_length, brav_mat, brav_inv, vec1, vec2, vec3)
        self.train()

    def opt_hyper(self):
        """
        Optimize hyperparameters by minimizing minus log likelihood w/ Nelder-Mead
        """
        args = (self.training_data['symm_norm'], self.training_data['forces_norm'])

        # initial guess
        x0 = np.array([self.sigma, self.length_scale])

        # nelder-mead opt
        res = minimize(minus_like_hyp, x0, args, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

        self.sigma, self.length_scale = res.x[0], res.x[1]

    def train(self):
        """
        Train ML model on training_data/ training_labels
        """
        if self.sklearn:
            self.model.fit(self.training_data['symm_norm'], self.training_data['forces_norm'])

        else:
            # optimize hyperparameters
            self.opt_hyper()

            # following: Algorithm 2.1 (pg. 19) of "Gaussian Processes for Machine Learning" by Rasmussen and Williams.
            self.K, self.L = get_SE_K(self.training_data['symm_norm'], self.sigma, self.length_scale)

            # get alpha and likelihood
            self.alpha = GP_SE_alpha(self.K, self.L, self.training_data['symm_norm'])

    def predict(self, structure, target='f'):
        """
        Predict on test data
        """
        if target == 'f':

            if self.sklearn:

                # TODO: check this
                self.pred, std = self.model.predict(self.test_data['symm_norm'], return_std=True)
                self.pred_var = std ** 2

            else:

                # TODO: correct this
                self.pred, self.pred_var = GP_SE_pred(self.test_data['symm_norm'], self.test_data['forces_norm'],
                                                      self.K, self.L, self.alpha, self.sigma,
                                                      self.length_scale, self.test_data)
        elif target == 'e':
            raise ValueError("Not implemented yet. Stay tuned.")

        else:
            raise ValueError("No proper ML target defined.")

    def is_var_inbound(self):
        return self.err_thresh > self.pred_var


class KernelRidgeRegression(RegressionModel):
    """KRR Regression Model"""

    def __init__(self, training_data, training_labels, test_data, test_labels, kernel,
                 alpha_range, gamma_range, cv, correction_folder, training_folder, sklearn, verbosity):
        """
        Initialization
        """
        self.alpha_range = alpha_range
        self.gamma_range = gamma_range
        self.kernel = kernel
        self.verbosity = verbosity
        self.sklearn = sklearn
        self.cv = cv

        if self.sklearn:
            self.model = GridSearchCV(KernelRidge(kernel=self.kernel), cv=self.cv,
                                      param_grid={"alpha": self.alpha_range,
                                                  "gamma": self.gamma_range})

        else:
            # PyFly implementation of Kernel Ridge Regression
            pass

        RegressionModel.__init__(self, model=self.model, training_data=training_data, test_data=test_data,
                                 correction_folder=correction_folder, training_folder=training_folder,
                                 model_type='krr', target='f', verbosity=verbosity)

    def train(self):
        """
        Train ML model on training_data/ training_labels
        """
        self.model.fit(self.training_data['symm_norm'], self.training_data['forces_norm'])

    def predict(self):
        """
        Predict on test data
        """
        self.model.predict(self.test_data['symm_norm'])
