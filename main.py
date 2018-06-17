#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name

"""" Production code -- Adaptive Machine-Learning Molecular Dynamics using Gaussian Processes

    # References:
        [1] Brockherde et al. Bypassing the Kohn-Sham equations with machine learning. Nature Communications 8, 872 (2017)
        [2] http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
        [3] http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py

Simon Batzner, Steven Torrisi, Jon Vandermause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from utility import Atom, MD_engine, ESPRESSO_config


def set_scf(arguments):
    """"
    Set constants used for SCF
    """

    # system params
    global ecut, nk, dim, config, alat
    ecut = 18.0  # plane wave cutoff energy
    nk = 8  # size of kpoint grid
    dim = 4  # size of supercell

    config = ESPRESSO_config(molecule=False, ecut=ecut, nk=nk, system_name=arguments.system_name)

    alat = 5.431  # lattice parameter of si in angstrom
    nat = 2 * dim ** 3
    memory = 1000

    # run params
    pref = arguments.system_name
    in_name = ".".join([arguments.system_name, 'scf', 'in'])
    out_name = ".".join([arguments.system_name, 'scf', 'out'])
    sh_name = ".".join([arguments.system_name, 'sh'])
    partition = arguments.partition

    if partition == 'kozinsky':
        pseudo_dir = '/n/home03/jonpvandermause/qe-6.2.1/pseudo'
        outdir = '/n/home03/jonpvandermause/Cluster/Si_Supercell_SCF'
        pw_loc = '/n/home03/jonpvandermause/qe-6.2.1/bin/pw.x'
        email = 'jonathan_vandermause@g.harvard.edu'

    elif partition == 'kaxiras':
        pseudo_dir = '/Users/steven/Documents/Schoolwork/CDMAT275/ESPRESSO/qe-6.0'
        outdir = ''
        pw_loc = ''
        email = 'torrisi@g.harvard.edu'

    elif partition == 'mit':
        pseudo_dir = ''
        outdir = ''
        pw_loc = ''
        email = 'sbatzner@mit.edu'

    elif partition == 'simon_local':
        pseudo_dir = '/Users/simonbatzner1/QE/qe-6.0/pseudo'
        outdir = sys.path.append(os.environ['ML_HOME'] + '/runs')
        pw_loc = '/Users/simonbatzner1/QE/qe-6.0/bin/pw.x'
        email = 'sbatzner@mit.edu'

    else:
        raise ValueError('Please provide a proper partition')
        sys.exit(1)

    return


def load_data(str_pref, sim_no):
    """"
    Load DFT data, set up input/target and convert to Atom representation
    """
    print("\nLoading data ...")
    pos = []
    ens = []

    for n in range(sim_no):
        # load arrays
        en_curr = np.reshape(np.load(str_pref + 'en_store/energy' + str(n) + '.npy'), 1)[0]
        pos_curr = np.load(str_pref + 'pos_store/pos' + str(n) + '.npy')

        # store arrays
        ens.append(en_curr)
        pos_curr = pos_curr.flatten()
        pos.append(pos_curr)

    print(pos[0])
    print(pos[0].shape)
    print("Number of training points: {}".format(len(pos)))

    # convert to np arrays
    pos = np.reshape(pos, (sim_no, pos[0].shape[0]))
    ens = np.reshape(ens, (sim_no, 1))

    alat = 4.10
    Npos = 15
    Atom1 = Atom(position=pos[Npos][:3] * alat, element='Al')
    Atom2 = Atom(position=pos[Npos][3:6] * alat, element='Al')
    Atom3 = Atom(position=pos[Npos][6:9] * alat, element='Al')
    Atom4 = Atom(position=pos[Npos][9:] * alat, element='Al')
    atoms = [Atom1, Atom2, Atom3, Atom4]

    return pos, ens, atoms


def build_gp(length_scale, length_scale_min, length_scale_max, verbosity):
    """
    Build Gaussian Process using scikit-learn, print hyperparams and return model
    :return: gp model
    """
    kernel = RBF(length_scale=length_scale, length_scale_bounds=(length_scale_min, length_scale_max))

    if verbosity:
        print("\n================================================================")
        print("\nKernel: {}".format(kernel))
        print("Hyperparameters: \n")

        for hyperparameter in kernel.hyperparameters:
            print(hyperparameter)
        print("Parameters:\n")

        params = kernel.get_params()
        for key in sorted(params):
            print("%s : %s" % (key, params[key]))
        print("\n================================================================")

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)

    return gp


def main(arguments):
    """
    Run adaptive MD using GP and uncertainty estimates
    """

    # params
    # str_pref = os.environ['PROJDIR']+'/Aluminium_Dataset/Store/'
    str_pref = arguments.data_dir
    sim_no = 201  # total number of data points

    # define scf params
    set_scf(arguments=arguments)

    # set up data
    x_train, y_train, atoms = load_data(str_pref, sim_no)

    # build gaussian process model
    gp = build_gp(length_scale=arguments.length_scale, length_scale_min=arguments.length_scale_min,
                  length_scale_max=arguments.length_scale_max, verbosity=arguments.verbosity)

    # train model
    print("\nFitting GP...")
    gp.fit(x_train, y_train)

    # build MD engine
    GP_engine = MD_engine(cell=alat * np.eye(3), input_atoms=atoms, ML_model=gp, model='GP',
                          store_trajectory=True, espresso_config=config, verbosity=arguments.verbosity,
                          assert_boundaries=False, dx=.1,
                          fd_accuracy=4,
                          threshold=0.1)

    # run MD engine
    print("\nRunning MD engine...")
    GP_engine.run(1, .1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--partition', type=str, default='kozinsky')
    parser.add_argument('--system_name', type=str, default='system')
    parser.add_argument('--data_dir', type=str, default='.', help='directory where training data are located')
    parser.add_argument('--length_scale', type=float, default=10, help='length-scale of Gaussian Process')
    parser.add_argument('--length_scale_min', type=str, default=1e-2,
                        help='minimum of range of hyperparams for GP length-scale to optimize')
    parser.add_argument('--length_scale_max', type=str, default=1e2,
                        help='maximum of range of hyperparams for GP length-scale to optimize')
    parser.add_argument('--verbosity', type=int, default=5, help='1 to 5')

    args = parser.parse_args()
    print("\nArguments: ", args, "\n")

    global pref, pseudo_dir, outdir, alat, ecut, nk, dim, nat, pw_loc, in_name, out_name, sh_name, partition, memory, email, config
    main(arguments=args)
