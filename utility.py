#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name

"""" Helper Classes and Functions

Simon Batzner, Steven Torrisi, Jon Vandermause
"""

import os

import numpy as np
import numpy.random as rand
import numpy.linalg as la
from ase import Atoms

from util.project_objects import *
from util.project_pwscf import *
from Solid_State import KRR_reproduce
from Solid_State import KRR_Functions


class MD_engine():
    def __init__(self, input_atoms=[], cell=np.eye(3), dx=.1, verbosity=1, model='SHO', store_trajectory=True,
                 espresso_config=None, thermo_config=None, ML_model=None, assert_boundaries=True, fd_accuracy=2,
                 threshold=0):
        """
        Initialize the features of the system, which include:
        input_atoms: Atoms in the unit cell, list of objects of type Atom (defined later on in this notebook)
        cell:  3x3 matrix [[a11 a12 a13],..] which define the dimensions of the unit cell.
        dx:    Perturbation distance used to evaluate the forces by finite differences in potential energy.
        verbosity: integer from 0 to 5 which determines how much information will be printed about runs.
                    Ranging from 0 as silent to 5 is TMI and is mostly for debugging.
        model: The energy model used.
        store_trajectory: Boolean which determines if positions will be saved after every time step.
        epsresso_config: Espresso_config object which determines how on-the-fly quantum espresso runs will be
                        parameteried.
        thermo_config:   Thermo config object which determines the thermostate methodology that will be used
                        for molecular dynamics.
        ML_model: Object which parameterizes a ML model which returns energies when certain configurations are input.
        assert_boundaries : Determines if boundary conditions are asserted at the end of each position update
                            (which forces atoms to lie within the unit cell).
        fd_accuracy: Integer 2 or 4. Determines if the finite difference force calculation will be evaluated to
                        second or fourth order in dx. Note that this increases the number of energy calculations
                        and thus increases run time.
        threshold: Defines an uncertainty cutoff above which a DFT run will be called.
        """
        self.verbosity = verbosity

        self.atoms = input_atoms  # Instantiate internal list of atoms, populate via helper function
        for atom in input_atoms:
            if self.verbosity == 5:
                print('Loading in', atom)

        self.dx = dx
        self.cell = np.array(cell)
        self.model = model

        self.time = 0.

        self.store_trajectory = store_trajectory

        # Construct trajectories object
        if store_trajectory:
            self.trajs = []
            for n in range(len(self.atoms)):
                self.trajs.append([])

        # Set configurations
        self.espresso_config = espresso_config or None
        self.thermo_config = thermo_config or None
        self.ML_model = ML_model or None

        self.assert_boundaries = assert_boundaries

        self.fd_accuracy = fd_accuracy

        self.threshold = threshold

    def add_atom(self, atom):
        """
        Helper function which adds atoms to the simulation
        """
        if self.verbosity == 4:
            print("Adding atom", atom)
        self.atoms.append(atom)

    def get_config_energy(self, atoms):
        """
        The positions of the atoms are passed to a given model of choice,
        in which the positions are converted into the configuration which is
        appropriate to be parsed by the model.

        For instance, the positions may need to be converted into
        atom-centered symmetry functions for the Gaussian Process model. By
        passing in the positions, the necessary mapping is performed, and then
        the appropriate model is called.
        """

        # Simple Harmonic Oscillator
        if self.model == 'SHO':
            return SHO_energy(atoms)

        # Gaussian Process
        if self.model == 'GP':
            config = GP_config(self.atoms, self.cell)
            energy, sigma = GP_energy(config, self.ML_model)

            # Check sigma bound
            if la.norm(sigma) > .01:
                print("Warning: Uncertainty cutoff reached: sigma = {}".format(sigma))

            return energy

        # Kernel Ridge Regression
        if self.model == "KRR":
            # Helper function which converts atom positions into a configuration
            #  appropriate for the KRR model. Could then be modified
            #  later depending on alternate representations.
            config = KRR_config(self.atoms, self.cell)
            return KRR_energy(config, self.ML_model)

        # Lennard Jones
        if self.model == 'LJ':
            return LJ_energy(atoms)

    def update_atom_forces(self):
        """
        Perturbs the atoms by a small amount dx in each direction
        and returns the gradient (and thus the force)
        via a finite-difference approximation.

        Or, if the engine's model is ab-initio molecular dynamics,
        runs quantum ESPRESSO on the current configuration and then
        stores the forces output from ESPRESSO as the atomic forces.
        """

        if self.model == 'AIMD':
            results = run_espresso(self.atoms, self.cell, qe_config=self.espresso_config)

            if self.verbosity == 4: print("E0:", results['energy'])

            force_list = results['forces']
            for n in range(len(force_list)):
                self.atoms[n].force = list(np.array(results['forces'][n]) * 13.6 / 0.529177)

            return

        E0 = self.get_config_energy(self.atoms)

        if self.verbosity == 4: print('E0:', E0)
        atoms = self.atoms

        # Depending on the finite difference accuracy specified at instantiation,
        # perform either 2 or 4 potential energy calculations.

        # Second-order finite-difference accuracy
        if self.fd_accuracy == 2:
            for atom in atoms:
                for coord in range(3):
                    # Perturb the atom to the E(x+dx) position
                    atom.position[coord] += self.dx
                    Eplus = self.get_config_energy(atoms)
                    # Perturb the atom to the E(x-dx) position
                    atom.position[coord] -= 2 * self.dx
                    Eminus = self.get_config_energy(atoms)

                    # Return atom to initial position
                    atom.position[coord] += self.dx

                    atom.force[coord] = -first_derivative_2nd(Eminus, Eplus, self.dx)
                    if self.verbosity == 5:
                        print("Just assigned force on atom's coordinate", coord, " to be ",
                              -first_derivative_2nd(Eminus, Eplus, self.dx))

        # Fourth-order finite-difference accuracy
        if self.fd_accuracy == 4:
            for atom in atoms:
                for coord in range(3):
                    # Perturb the atom to the E(x+dx) position
                    atom.position[coord] += self.dx
                    Eplus = self.get_config_energy(atoms)
                    # Perturb the atom to the E(x+2dx) position
                    atom.position[coord] += self.dx
                    Epp = self.get_config_energy(atoms)
                    # Perturb the atom to the E(x-2dx) position
                    atom.position[coord] -= 4.0 * self.dx
                    Emm = self.get_config_energy(atoms)
                    # Perturb the atom to the E(x-2dx) position
                    atom.position[coord] += self.dx
                    Eminus = self.get_config_energy(atoms)

                    atom.position[coord] += self.dx

                    atom.force[coord] = -first_derivative_4th(Emm, Eminus, Eplus, Epp, self.dx)
                    if self.verbosity == 5:
                        print("Just assigned force on atom's coordinate", coord, " to be ",
                              -first_derivative_2nd(Eminus, Eplus, self.dx))

        atom.apply_constraint()

    def take_timestep(self, dt, method='Verlet'):
        """
        Propagate the atoms forward by timestep dt according to it's current force and previous position.
        Note that the first time step does not have the benefit of a previous position, so, we use
        a standard third-order Euler timestep incorporating information about the velocity and the force.
        """
        temp_num = 0.

        # Third-order euler method
        # Not the worst way to begin a Verlet run, see:
        # https://en.wikipedia.org/wiki/Verlet_integration#Starting_the_iteration

        if method == 'TO_Euler':
            for atom in self.atoms:
                for coord in range(3):
                    atom.prev_pos[coord] = np.copy(atom.position[coord])
                    atom.position[coord] += atom.velocity[coord] * dt + atom.force[coord] * dt ** 2 / atom.mass
                    atom.velocity[coord] += atom.force[coord] * dt / atom.mass

        ######
        # Superior Verlet integration
        # Citation:  https://en.wikipedia.org/wiki/Verlet_integration
        ######
        elif method == 'Verlet':
            for atom in self.atoms:
                for coord in range(3):
                    # Store the current position to later store as the previous position
                    # After using it to update the position

                    temp_num = np.copy(atom.position[coord])
                    atom.position[coord] = 2 * atom.position[coord] - atom.prev_pos[coord] + atom.force[
                                                                                                 coord] * dt ** 2 / atom.mass
                    atom.velocity[coord] += atom.force[coord] * dt / atom.mass
                    atom.prev_pos[coord] = np.copy(temp_num)
                    if self.verbosity == 5: print("Just propagated a distance of ",
                                                  atom.position[coord] - atom.prev_pos[coord])

        if self.store_trajectory:
            for n in range(len(self.atoms)):
                self.trajs[n].append(np.copy(self.atoms[n].position))
        self.time += dt

    def run(self, tf, dt):
        """
        Handles timestepping; at each step, calculates the force and then
        advances via the take_timestep method.

        """

        # The very first timestep often doesn't have the 'previous position' to use,
        # so the third-order Euler method starts us off using information about the position,
        # velocity (if provided) and force:

        if self.time == 0:
            self.update_atom_forces()
            self.take_timestep(dt, method='TO_Euler')
            # Assert boundary conditions (push all atoms to unit cell) if specified
            if self.assert_boundaries: self.assert_boundary_conditions()

            if self.verbosity >= 3:
                self.print_positions()

        while self.time < tf:
            self.update_atom_forces()
            self.take_timestep(dt)
            if self.assert_boundaries: self.assert_boundary_conditions()

            if self.verbosity >= 3:
                self.print_positions()

            # check uncertainty and retrain model if it exceeds specified threshold
            if (self.model == 'GP' or self.model == 'KRR') and self.threshold > 0:

                valid = self.gauge_uncertainty()

                if valid:
                    if self.verbosity == 5:
                        print("Uncertainty valid")
                    continue

                # move to previous md step, compute DFT, update training set, retrain ML model
                else:
                    self.take_timestep(-dt)
                    self.time -= dt
                    run_espresso(self.atoms, self.cell, qe_config=self.espresso_config,
                                 iscorrection=True)
                    self.retrain_ml_model()

    def assert_boundary_conditions(self):
        """
        We seek to have our atoms be entirely within the unit cell, that is,
        for bravais lattice vectors a1 a2 and a3, we want our atoms to be at positions

         x= a a1 + b a2 + c a3
         where a, b, c in [0,1)

         So in order to impose this condition, we invert the matrix equation such that

          [a11 a12 a13] [a] = [x1]
          [a21 a22 a23] [b] = [x2]
          [a31 a32 a33] [c] = [x3]

          And if we find that a, b, c not in [0,1) we modulo by 1.
        """
        a1 = self.cell[0]
        a2 = self.cell[1]
        a3 = self.cell[2]

        for atom in self.atoms:

            coords = np.dot(la.inv(self.cell), atom.position)
            if self.verbosity == 5:
                print('Atom positions before BC check:', atom.position)
                print('Resultant coords:', coords)

            if any([coord > 1.0 for coord in coords]) or any([coord < 0.0 for coord in coords]):
                if self.verbosity == 4: print('Atom positions before BC check:', atom.position)

                atom.position = a1 * (coords[0] % 1) + a2 * (coords[1] % 1) + a3 * (coords[2] % 1)
                if self.verbosity == 4: print("BC updated position:", atom.position)

    def print_positions(self, forces=True):

        print("T=", self.time)
        for n in range(len(self.atoms)):

            pos = self.atoms[n].position
            force = self.atoms[n].force
            if forces:
                print('Atom %d:' % n, np.round(pos, decimals=4), ' Force:', np.round(force, decimals=6))
            else:
                print('Atom %d:' % n, np.round(pos, decimals=4))

    def gauge_uncertainty(self):
        """
        For later implementation with the Gaussian Process model.
        Will check to see if the uncertainty of the model's prediction of the energy
        within a given configuration is within an acceptable bound given by threshold.

        If it is unacceptable, the current configuration is exported into a Quantum ESPRESSO run,
        which is then added into the ML dataset and the model is re-trained.
        """
        if self.model == "KRR":
            pass
            #
            # config = KRR.config(self.atoms, self.cell)
            # if self.threshold < KRR_uncertainty(configuration):
            #     print("!! The uncertainty of the model is outside of the ")
            #     return False

        if self.model == "GP":

            config = GP_config(self.atoms, self.cell)
            energy, sigma = GP_energy(config, self.ML_model)

            if self.threshold < sigma:
                print("CAUTION: The uncertainty of the model is outside of the specified threshold")
                return False

        return True

    def retrain_ml_model(self, model=None, ML_model=None):
        """"
        If uncertainty has surpassed cutoff value, update training set and retrain ML model
        """

        # model = model or self.model
        # ML_model = ML_model or self.ML_model

        if self.model == 'KRR':
            # init_train_set = self.ML_model.original_train_set
            # init_train_ens = self.ML_model.original_train_ens
            #
            # aug_train_set, aug_train_ens = get_aug_values(self.espresso_config.correction_folder)
            #
            # new_train_set = np.concatenate(init_train_set, aug_train_set)
            # new_train_ens = np.concatenate(init_train_ens, aug_train_ens)
            #
            # self.ML_model.quick_fit(new_train_set, new_train_ens, self.ML_model.alpha, self.ML_model.gamma)
            # return
            pass

        if self.model == "GP":
            # update training set
            x_init = self.ML_model.original_train_set
            y_init = self.ML_model.original_train_ens

            x_add, y_add = get_aug_values(self.espresso_config.correction_folder)

            x_upd = np.concatenate(x_init, x_add)
            y_upd = np.concatenate(y_init, y_add)

            # retrain
            self.ML_model.fit(x_upd, y_upd)

            return


class Atom():
    mass_dict = {'H': 1.0, "Al": 26.981539, "Si": 28.0855}

    def __init__(self, position=[0., 0., 0.], velocity=[0., 0., 0.], force=[0., 0., 0.], initial_pos=[0, 0, 0],
                 mass=None,
                 element='', constraint=[False, False, False]):

        self.position = np.array(position)
        # Used in Verlet integration
        self.prev_pos = np.array(self.position)
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.element = element

        self.constraint = constraint

        self.fingerprint = rand.rand()  # This is how I tell atoms apart. Easier than indexing them manually...

        if self.element in self.mass_dict.keys() and mass == None:
            self.mass = self.mass_dict[self.element]
        else:
            self.mass = mass or 1.0

        ## Used for testing with a simple harmonic oscillator potential
        ## in which the force is merely the displacement squared from initial position
        self.initial_pos = np.array(initial_pos)

        self.parameters = {'position': self.position,
                           'velocity': self.velocity,
                           'force': self.force,
                           'mass': self.mass,
                           'element': self.element,
                           'constraint': self.constraint,
                           'initial_pos': self.initial_pos}

    def __str__(self):
        return str(self.parameters)

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_force(self):
        return self.force

    def apply_constraint(self):
        for n in range(3):
            if self.constraint[n]:
                self.velocity[n] = 0.
                self.force[n] = 0.


class ESPRESSO_config(object):
    def get_correction_number(self):
        folders_in_correction_folder = list(os.walk(self.correction_folder))[0][1]

        steps = [fold for fold in folders_in_correction_folder if self.system_name + "_step_" in fold]

        if len(steps) >= 1:
            stepvals = [int(fold.split('_')[-1]) for fold in steps]
            correction_number = max(stepvals)
        else:
            return 0
        return correction_number + 1

    def __init__(self, workdir=None, run_pwscf=None, pseudopotentials=None,
                 molecule=False, nk=15, correction_folder=None, system_name="", correction_number=0, ecut=40, ):

        # Where to store QE calculations
        self.workdir = os.environ['PROJDIR'] + '/AIMD'
        # Runs the PWSCF
        self.run_pwscf = os.environ['PWSCF_COMMAND']
        # Helpful dictionary of pseudopotential objects
        self.pseudopotentials = {"H": PseudoPotential(path=os.environ["ESPRESSO_PSEUDO"], ptype='uspp', element='H',
                                                      functional='GGA', name='H.pbe-kjpaw.UPF'),
                                 "Si": PseudoPotential(path=os.environ["ESPRESSO_PSEUDO"], ptype='?', element='Si',
                                                       functional='?', name='Si.pz-vbc.UPF'),
                                 'Al': PseudoPotential(path=os.environ["ESPRESSO_PSEUDO"], ptype='upf', element='Al',
                                                       functional='PZ', name='Al.pz-vbc.UPF')}
        # Periodic boundary conditions and nk X nk X nk k-point mesh,
        # or no PBCs and Gamma k point only
        self.molecule = molecule

        # Dimensions of k point grid
        self.nk = nk

        # Will be used for correction folders later

        self.system_name = system_name

        self.correction_folder = correction_folder or self.workdir

        self.correction_number = self.get_correction_number()

        self.ecut = ecut


# set to None for simon running tests locally, else use both lines
# config2 = ESPRESSO_config(molecule=False)
# config2.molecule = False
config2 = None


def first_derivative_2nd(fm, fp, h):
    """
    Computes the second-order accurate finite difference form of the first derivative
    which is (  fp/2 - fm/2)/(h)
    as seen on Wikipedia: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """
    if h == 0:
        print("Warning... Trying to divide by zero. Derivative will diverge.")
    return (fp - fm) / float(2 * h)


def first_derivative_4th(fmm, fm, fp, fpp, h):
    """
    Computes the fourth-order accurate finite difference form of the first derivative
    which is (fmm/12  - 2 fm /3 + 2 fp /3 - fpp /12)/h
    as seen on Wikipedia: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """

    if h == 0:
        print("Warning... Trying to divide by zero. Derivative will diverge.")

    return (fmm / 12. - 2 * fm / 3. + 2 * fp / 3. - fpp / 12.) / float(h)


def run_espresso(atoms, cell, qe_config=config2, ecut=40, molecule=True, stepcount=0, iscorrection=False):
    pseudopots = {}
    elements = [atom.element for atom in atoms]
    for element in elements:
        pseudopots[element] = qe_config.pseudopotentials[element]

    ase_struc = Atoms(symbols=[atom.element for atom in atoms],
                      positions=[atom.position for atom in atoms],
                      cell=cell,
                      pbc=[0, 0, 0] if molecule else [1, 1, 1])

    struc = Struc(ase2struc(ase_struc))

    if qe_config.molecule:
        kpts = Kpoints(gridsize=[1, 1, 1], option='gamma', offset=False)
    else:
        nk = qe_config.nk
        kpts = Kpoints(gridsize=[nk, nk, nk], option='automatic', offset=False)

    if iscorrection:
        dirname = 'step_' + str(stepcount)
    else:
        dirname = 'temprun'
    runpath = Dir(path=os.path.join(os.environ['PROJDIR'], "AIMD", dirname))
    input_params = PWscf_inparam({
        'CONTROL': {
            'prefix': 'AIMD',
            'calculation': 'scf',
            'pseudo_dir': os.environ['ESPRESSO_PSEUDO'],
            'outdir': runpath.path,
            #            'wfcdir': runpath.path,
            'disk_io': 'low',
            'tprnfor': True,
            'wf_collect': False
        },
        'SYSTEM': {
            'ecutwfc': ecut,
            'ecutrho': ecut * 8,
            #           'nspin': 4 if 'rel' in potname else 1,

            'occupations': 'smearing',
            'smearing': 'mp',
            'degauss': 0.02,
            # 'noinv': False
            # 'lspinorb':True if 'rel' in potname else False,
        },
        'ELECTRONS': {
            'diagonalization': 'david',
            'mixing_beta': 0.5,
            'conv_thr': 1e-7
        },
        'IONS': {},
        'CELL': {},
    })

    output_file = run_qe_pwscf(runpath=runpath, struc=struc, pseudopots=pseudopots,
                               params=input_params, kpoints=kpts,
                               ncpu=1)
    output = parse_qe_pwscf_output(outfile=output_file)

    print("Parsed QE output file... ")

    with open(runpath.path + 'en', 'w') as f:
        f.write(str(output['energy']))
    with open(runpath.path + 'pos', 'w')as f:
        for pos in [atom.position for atom in atoms]:
            f.write(str(pos) + '\n')

    return output


def SHO_config(atoms):
    # No special configuration needed
    return [atom.position for atom in atoms]


def SHO_energy(atoms, kx=10.0, ky=1.0, kz=1.0):
    """
    atoms: list of Atom objects
    kx, ky, kz: possibility for anisotropic SHO, can accept lists (which should be
    equal to number of atoms) or float, which then will apply equally to all atoms
    """
    if type(kx) is list:
        if len(kx) != len(atoms):
            print("Warning! Number of Kx supplied is not equal to number of atoms")
    elif type(kx) is float:
        kx = [kx] * len(atoms)

    if type(ky) is list:
        if len(ky) != len(atoms):
            print("Warning! Number of Ky supplied is not equal to number of atoms")
    elif type(ky) is float:
        ky = [ky] * len(atoms)

    if type(kz) is list:
        if len(kz) != len(atoms):
            print("Warning! Number of Kz supplied is not equal to number of atoms")
    elif type(kz) is float:
        kz = [kz] * len(atoms)

    init_pos = [atom.initial_pos for atom in atoms]
    positions = [atom.position for atom in atoms]

    K = [kx, ky, kz]

    # Compute the energy
    E = 0
    for m in range(len(atoms)):
        for n in range(3):
            E += K[n][m] * (init_pos[m][n] - positions[m][n]) ** 2

    return E


def KRR_config(atoms, cell):
    coords = np.empty(shape=(3, len(atoms)))
    # print("Here are the before positions:",[atom.position for atom in atoms])
    for n in range(len(atoms)):
        coords[:, n] = np.dot(la.inv(cell), atoms[n].position)
    # print("And the after:",coords.T.flatten().reshape(1,-1))
    return coords.T.flatten().reshape(1, -1)


#
# # Leaving it this way for now because it's simpler for testing aluminum plus no
# #   risk of matrix inversion introducing numerical noise
# def KRR_config(atoms, cell):
#     alat = 4.10
#     # print("Here are the before positions:",[atom.position for atom in atoms])
#
#     coords = np.empty(shape=(3, len(atoms)))
#     for n in range(len(atoms)):
#         coords[:, n] = atoms[n].position / alat
#     # print("And the after:",coords.T.flatten().reshape(1,-1))
#     return coords.T.flatten().reshape(1, -1)


def LJ_energy(atoms, rm=.5, eps=10., debug=False):
    """
    Ordinary 12-6 form of the Lennard Jones potential
    """
    E = 0.
    for at in atoms:
        for at2 in atoms:
            if at.fingerprint != at2.fingerprint:

                disp = la.norm(at.position - at2.position)
                if debug:
                    print('Current LJ disp between atoms is:', disp)

                E += .5 * eps * ((rm / disp) ** 12 - 2 * (rm / disp) ** 6)

    return E


def KRR_energy(krrconfig, model):
    return model.predict(krrconfig)


def GP_config(atoms, cell):
    coords = np.empty(shape=(3, len(atoms)))

    for n in range(len(atoms)):
        coords[:, n] = np.dot(la.inv(cell), atoms[n].position)

    coords = coords.flatten()
    coords = coords.reshape(1, -1)
    print(coords.shape)

    return coords


def GP_energy(gpconfig, model):
    return model.predict(gpconfig, return_std=True)


def get_aug_values(correction_folder, keyword='step', ML_model=None):
    energies = []
    positions = []
    forces = []
    indices = []
    if len([folder for folder in list(os.walk(correction_folder))[0][1]]) > 0:
        for fold in [folder for folder in list(os.walk(correction_folder))[0][1] if keyword in folder]:
            index = int(fold.split("_")[-1])
            if ML_model != None:
                if index not in ML_model.aug_indices:
                    ML_model.aug_indices.append(index)
                else:
                    continue
            fold = correction_folder + '/' + fold
            print(fold)
            with open(fold + '/en', 'r') as f:
                energies.append(float(f.readlines()[0]))

            with open(fold + '/pos', 'r') as f:
                read_pos = f.readlines()
                for n in range(len(read_pos)):
                    curr_pos = read_pos[n].strip().strip('[').strip('\n').strip(']')
                    print(curr_pos)
                    curr_pos = [float(x) for x in curr_pos.split()]
                    for x in curr_pos:
                        positions.append(x)

            print("Found positions", positions, "and energies", energies)

            return positions, energies
