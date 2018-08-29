#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""" Gaussian Process Regression model

Implementation is based on Algorithm 2.1 (pg. 19) of
"Gaussian Processes for Machine Learning" by Rasmussen and Williams

Simon Batzner
"""
import os

import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from ..two_body import two_body
from otf import parse_qe_input, parse_qe_forces, Structure


def get_outfiles(root_dir, out=True):
    """
    Find all files matching *.out OR *.in

    :param root_dir:    (str), dir to walk from
    :param out:         (bool), whether to look for .out (true) or .in files (false)
    :return: matching   (list), files in root_dir ending with .out
    """
    matching = []

    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:

            if out:
                if filename.endswith('.out'):
                    matching.append(os.path.join(root, filename))

            else:
                if filename.endswith('.in'):
                    matching.append(os.path.join(root, filename))

    return matching



# given list of Cartesian coordinates, return list of atomic environments
def get_cutoff_vecs(vec, brav_mat, brav_inv, vec1, vec2, vec3, cutoff):
    # get bravais coefficients
    coeff = np.matmul(brav_inv, vec)

    # get bravais coefficients for atoms within one super-super-cell
    coeffs = [[], [], []]
    for n in range(3):
        coeffs[n].append(coeff[n])
        coeffs[n].append(coeff[n] - 1)
        coeffs[n].append(coeff[n] + 1)
        coeffs[n].append(coeff[n] - 2)
        coeffs[n].append(coeff[n] + 2)

    # get vectors within cutoff
    vecs = []
    dists = []
    for m in range(len(coeffs[0])):
        for n in range(len(coeffs[1])):
            for p in range(len(coeffs[2])):
                vec_curr = coeffs[0][m] * vec1 + coeffs[1][n] * vec2 + coeffs[2][p] * vec3

                dist = np.linalg.norm(vec_curr)

                if dist < cutoff:
                    vecs.append(vec_curr)
                    dists.append(dist)

    return vecs, dists


# given list of cartesian coordinates, get chemical environment of specified atom
# pos = list of cartesian coordinates
# typs = list of atom types
def get_env_struc(pos, typs, atom, brav_mat, brav_inv, vec1, vec2, vec3, cutoff):
    pos_atom = np.array(pos[atom]).reshape(3, 1)
    typ = typs[atom]
    env = {'central_atom': typ, 'dists': [], 'xs': [], 'ys': [], 'zs': [],
           'xrel': [], 'yrel': [], 'zrel': [], 'types': []}

    # loop through positions to find all atoms and images in the neighborhood
    for n in range(len(pos)):
        # position relative to reference atom
        diff_curr = np.array(pos[n]).reshape(3, 1) - pos_atom

        # get images within cutoff
        vecs, dists = get_cutoff_vecs(diff_curr, brav_mat,
                                      brav_inv, vec1, vec2, vec3, cutoff)

        for vec, dist in zip(vecs, dists):
            # ignore self interaction
            if dist != 0:
                # append distance
                env['dists'].append(dist)

                # append coordinate differences
                env['xs'].append(vec[0][0])
                env['ys'].append(vec[1][0])
                env['zs'].append(vec[2][0])

                # append relative coordinate differences
                env['xrel'].append(vec[0][0] / dist)
                env['yrel'].append(vec[1][0] / dist)
                env['zrel'].append(vec[2][0] / dist)

                # append atom type
                env['types'].append(typs[n])

    #env['trip_dict'] = get_trip_dict(env)

    return env


# given list of cartesian coordinates, return list of chemical environments
def get_envs(pos, typs, brav_mat, brav_inv, vec1, vec2, vec3, cutoff):
    envs = []
    for n in range(len(pos)):
        atom = n
        env = get_env_struc(pos, typs, atom, brav_mat, brav_inv, vec1, vec2, vec3, cutoff)
        envs.append(env)

    return envs


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
        self.training_data = np.empty(0,)
        self.training_labels = np.empty(0,)

        # initiate database
        self.init_db()

    def init_db(self, root_dir):
        """Initialize database from root directory containing training data"""

        for file in get_outfiles(root_dir=root_dir, out=False):
            positions, species, cell = parse_qe_input(file)

        for file in get_outfiles(root_dir=root_dir, out=True):
            forces = parse_qe_forces(file)

        self.training_data = np.asarray(get_envs(pos=positions, typs=['Si'], brav_mat=brav_mat, brav_inv=brav_inv,
                                                 vec1=vec1, vec2=vec2, vec3=vec3, cutoff=cutoff))
        self.training_labels = np.asarray(forces)

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


if __name__ == "__main__":

    global alat, brav_inv, brav_mat, vec1, vec2, vec3, cutoff

    # set crystal structure
    dim = 3
    alat = 4.344404578
    unit_cell = [[0.0, alat / 2, alat / 2], [alat / 2, 0.0, alat / 2], [alat / 2, alat / 2, 0.0]]
    unit_pos = [['Si', [0, 0, 0]], ['Si', [alat / 4, alat / 4, alat / 4]]]
    brav_mat = np.array([[0.0, alat / 2, alat / 2], [alat / 2, 0.0, alat / 2], [alat / 2, alat / 2, 0.0]]) * dim
    brav_inv = np.linalg.inv(brav_mat)

    # bravais vectors
    vec1 = brav_mat[:, 0].reshape(3, 1)
    vec2 = brav_mat[:, 1].reshape(3, 1)
    vec3 = brav_mat[:, 2].reshape(3, 1)
    cutoff = 4.5

    positions, species, cell = parse_qe_input('pwscf.in')
    forces = parse_qe_forces('pwscf.out')
    structure = Structure(positions, species, cell)

    gp = GaussianProcess(kernel=two_body)
    gp.train()
    gp.predict(structure)