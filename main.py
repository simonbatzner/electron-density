#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name

"""" Production code -- Adaptive Machine-Learning Molecular Dynamics using Gaussian Processes

    Example run config:

    --partition simon_local --verbosity 1 --data_dir /Users/simonbatzner1/Desktop/Research/Research_Code/ML-electron-density/Aluminium_Dataset/Store/ --system_name='al' --np 1

    Parameters
    ----------
    partition           str, default: 'kozinsky'    -- partition to run on
    data_dir            str, default: '.'           -- directory where training data are located
    system_name         str, default='Al'           -- name of materials system
    system_type         str, default='solid'        -- "solid" or "molecule"'
    kernel              str, default='rbf'          -- kernel used in gaussian process: "rbf", "matern", "c_rbf" or "expsinesquared"
    length_scale        float, default=10           -- length-scale of Gaussian Process
    length_scale_min    float, default=1e-3         -- minimum of range for length-scale
    length_scale_max    float, default=1e3          -- maximum of range for length-scale
    nu                  float, default=1.5          -- nu param in matern kernel
    const_val           type=float, default=1.      -- constant value for constant kernel
    const_val_min       float, default=1e-2         -- miminum of range for constant value for constant kernel
    const_val_max       float, default=1e2          -- maximum of range for constant value for constant kernel
    periodicity         float, default=1.           -- periodicity for expsinesquared kernel
    periodicity_min     float, default=1e-2         -- miminum of range for periodicity for expsinesquared kernel
    periodicity_max     float, default=1e2          -- maximum of range for periodicity for expsinesquared kernel')
    alat                float, default=4.10         -- lattice parameter
    np                  int, default=1              -- number of copies for MPI, 1 sets it to serial mode
    nk                  int, default=0              -- number of pools
    nt                  int, default=0              -- number of FFT task groups
    nd                  int, default=0              -- number of linear algebra groups
    verbosity           int, default=5              -- verbosity of help from silent (1) to debug config (5)


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
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel as C

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

    # debug config
    ecut = 5.0
    nk = 1
    dim = 1

    if arguments.system_type == "solid":
        config = ESPRESSO_config(molecule=False, ecut=ecut, nk=nk, system_name=arguments.system_name,
                                 parallelization={'np': arguments.np, 'nk': arguments.nk, 'nt': arguments.nt,
                                                  'nd': arguments.nd})
    elif arguments.system_type == "molecule":
        config = ESPRESSO_config(molecule=True, ecut=ecut, nk=nk, system_name=arguments.system_name,
                                 parallelization={'np': arguments.np, 'nk': arguments.nk, 'nt': arguments.nt,
                                                  'nd': arguments.nd})
    else:
        raise ValueError('Please provide a proper system type: molecule or solid')
        sys.exit(1)

    alat = arguments.alat
    nat = 2 * dim ** 3
    memory = 1000

    # run params
    pref = arguments.system_name
    in_name = ".".join([arguments.system_name, 'scf', 'in'])
    out_name = ".".join([arguments.system_name, 'scf', 'out'])
    sh_name = ".".join([arguments.system_name, 'sh'])
    partition = arguments.partition

    if partition == 'kozinsky':

        os.environ['PWSCF_COMMAND'] = '/n/home03/jonpvandermause/qe-6.2.1/bin/pw.x'
        os.environ["ESPRESSO_PSEUDO"] = '/n/home03/jonpvandermause/qe-6.2.1/pseudo'
        email = 'jonathan_vandermause@g.harvard.edu'

        # for future use
        # outdir = '/n/home03/jonpvandermause/Cluster/Si_Supercell_SCF'

    elif partition == 'kaxiras':
        os.environ['PWSCF_COMMAND'] = '/Users/simonbatzner1/QE/qe-6.0/bin/pw.x'
        os.environ["ESPRESSO_PSEUDO"] = '/Users/steven/Documents/Schoolwork/CDMAT275/ESPRESSO/qe-6.0'
        email = 'torrisi@g.harvard.edu'

    elif partition == 'mit':
        os.environ['PWSCF_COMMAND'] = ''
        os.environ["ESPRESSO_PSEUDO"] = ''

    elif partition == 'simon_local':
        os.environ['PWSCF_COMMAND'] = '/Users/simonbatzner1/QE/qe-6.0/bin/pw.x'
        os.environ["ESPRESSO_PSEUDO"] = '/Users/simonbatzner1/QE/qe-6.0/pseudo'

    else:
        raise ValueError('Please provide a proper partition')
        sys.exit(1)

    return


def load_data(str_pref, sim_no, arguments):
    """"
    Load DFT data, set up input/target data and convert to Atom representation
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

    print("Number of training points: {}".format(len(pos)))

    # convert to np arrays
    pos = np.reshape(pos, (sim_no, pos[0].shape[0]))
    ens = np.reshape(ens, (sim_no, 1))

    # starting config for MD engine
    n_atoms = pos.shape[1] // 3
    input_atoms = []

    for n in range(n_atoms):
        input_atoms.append(
            Atom(position=pos[10][(n * 3):(n * 3 + 3)] * alat, element=str(arguments.system_name.title().strip('"\''))))

    return pos, ens, input_atoms


def build_gp(arguments):
    """
    Build Gaussian Process using scikit-learn, print hyperparams and return model
    :return: gp model
    """

    kernel_dict = {'c_rbf': C(arguments.const_val, (arguments.const_val_min, arguments.const_val_max)) * RBF(
        length_scale=arguments.length_scale, length_scale_bounds=(
            arguments.length_scale_min, arguments.length_scale_max)),
                   'rbf': RBF(length_scale=arguments.length_scale,
                              length_scale_bounds=(arguments.length_scale_min, arguments.length_scale_max)),
                   'matern': Matern(length_scale=arguments.length_scale,
                                    length_scale_bounds=(arguments.length_scale_min, arguments.length_scale_max),
                                    nu=arguments.nu),
                   'expsinesquared': ExpSineSquared(length_scale=arguments.length_scale,
                                                    periodicity=arguments.periodicity,
                                                    length_scale_bounds=(
                                                        arguments.length_scale_min, arguments.length_scale_max),
                                                    periodicity_bounds=(
                                                        arguments.periodicity_min, arguments.periodicity_max))}

    kernel = kernel_dict[arguments.kernel]

    if arguments.verbosity:
        print("\n================================================================")
        print("\nKernel: {}".format(kernel))
        print("\nHyperparameters:")

        for hyperparameter in kernel.hyperparameters:
            print(hyperparameter)
        print("\nParameters:")

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

    # UPDATE THIS
    sim_no = 201  # total number of data points

    # define scf params
    set_scf(arguments=arguments)

    # set up data
    x_train, y_train, input_atoms = load_data(str_pref=str_pref, sim_no=sim_no, arguments=arguments)

    # build gaussian process model
    gp = build_gp(arguments)

    gp.original_train_set = x_train
    gp.original_train_ens = y_train

    # train model
    print("\nFitting GP...")
    gp.fit(x_train, y_train)

    # build MD engine
    GP_engine = MD_engine(cell=alat * np.eye(3), input_atoms=input_atoms, ML_model=gp, model='GP',
                          store_trajectory=True, qe_config=config, verbosity=arguments.verbosity,
                          assert_boundaries=False, dx=.1,
                          fd_accuracy=4,
                          uncertainty_threshold=0.1)

    # run MD engine
    print("\nRunning MD engine...\n")
    GP_engine.run(1, .1)


def parse_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--partition', type=str, default='kozinsky')
    parser.add_argument('--data_dir', type=str, default='.', help='directory where training data are located')
    parser.add_argument('--system_name', type=str, default='Al')
    parser.add_argument('--system_type', type=str, default='solid', help='"solid" or "molecule"')
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='GP kernel: "rbf", "matern", "c_rbf" or "expsinesquared"')
    parser.add_argument('--length_scale', type=float, default=10, help='length-scale of Gaussian Process')
    parser.add_argument('--length_scale_min', type=float, default=1e-3,
                        help='minimum of range for length-scale')
    parser.add_argument('--length_scale_max', type=float, default=1e3,
                        help='maximum of range for length-scale')
    parser.add_argument('--nu', type=float, default=1.5,
                        help='nu param in matern kernel')
    parser.add_argument('--const_val', type=float, default=1.,
                        help='constant value for constant kernel')
    parser.add_argument('--const_val_min', type=float, default=1e-2,
                        help='miminum of range for constant value for constant kernel')
    parser.add_argument('--const_val_max', type=float, default=1e2,
                        help='maximum of range for constant value for constant kernel')
    parser.add_argument('--periodicity', type=float, default=1.,
                        help='periodicity for expsinesquared kernel')
    parser.add_argument('--periodicity_min', type=float, default=1e-2,
                        help='miminum of range for periodicity for expsinesquared kernel')
    parser.add_argument('--periodicity_max', type=float, default=1e2,
                        help='maximum of range for periodicity for expsinesquared kernel')
    parser.add_argument('--alat', type=float, default=4.10)
    parser.add_argument('--verbosity', type=int, default=5, help='1 to 5')
    parser.add_argument('--np', type=int, default='1')
    parser.add_argument('--nk', type=int, default='0')
    parser.add_argument('--nt', type=int, default='0')
    parser.add_argument('--nd', type=int, default='0')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global pref, alat, ecut, nk, dim, nat, in_name, out_name, partition, config, verbosity

    arguments = parse_args()
    verbosity = arguments.verbosity
    main(arguments=arguments)
