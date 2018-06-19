#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name

"""" Helper Classes and Functions

Simon Batzner, Steven Torrisi, Jon Vandermause
"""

import numpy.random as rand
import numpy.linalg as la
from ase import Atoms

from util.project_objects import *
from util.project_pwscf import *


class MD_engine():
    def __init__(self, input_atoms=[], cell=np.eye(3), dx=.1, verbosity=1, model='SHO', store_trajectory=True,
                 qe_config=None, thermo_config=None, ML_model=None, assert_boundaries=True, fd_accuracy=2,
                 uncertainty_threshold=0, energy_or_force_driven='energy'):

        # @STEVEN: PLESE CHECK WHETHER I SPECIFIED PARAMS CORRECTLY IN DOCS
        """
        Parameters
        ----------

        input_atoms:            list, atoms in the unit cell, list of objects of type Atom
        cell:                   numpy array, 3x3 matrix [[a11 a12 a13],..] which define the dimensions of the unit cell
        dx:                     float, perturbation distance used to evaluate the forces by finite differences in potential energy
        verbosity:              integer, ranges from 0 to 5 which determines how much information will be printed about runs.
        model:                  string, energy model used
        store_trajectory:       boolean, if positions will be saved after every time step.
        qe_config:               Espresso_config object, determines how on-the-fly quantum espresso runs will be
                                parameterized
        thermo_config:          Thermo config object, determines the thermostate methodology that will be used
                                for molecular dynamics.
        ML_model:               ML model object, parameterizes a ML model which returns energies when certain configurations are input
        assert_boundaries :     boolean, if boundary conditions are asserted at the end of each position update
                                (which forces atoms to lie within the unit cell)
        fd_accuracy:            integer: 2 or 4, determines if the finite difference force calculation will be evaluated to
                                second or fourth order in dx. Note that this increases the number of energy calculations
                                and thus increases run time
        uncertainYthreshold     float, defines an uncertainty cutoff above which a DFT run will be called
        energy_or_force_driven: str, maps to 'driver': dictates if the finite difference energy model drives the simulation
                                or if the force comes directly from the model (such as with AIMD.)
        """

        self.verbosity = verbosity

        self.atoms = input_atoms

        for atom in input_atoms:
            if self.verbosity == 5:
                print('Loading in', atom)

        self.dx = dx
        self.cell = np.array(cell)
        self.model = model

        self.time = 0.

        self.store_trajectory = store_trajectory

        if store_trajectory:
            self.trajs = []
            for n in range(len(self.atoms)):
                self.trajs.append([])

        # config
        self.qe_config = qe_config or None
        self.thermo_config = thermo_config or None
        self.ML_model = ML_model or None

        self.assert_boundaries = assert_boundaries

        self.fd_accuracy = fd_accuracy

        self.uncertainty_threshold = None or uncertainty_threshold
        self.driver = energy_or_force_driven

    def get_config_energy(self, atoms=None):
        """
        Transforms atomic positions to configuration, evaluates with specified model
        """

        atoms = atoms or self.atoms

        # Simple Harmonic Oscillator
        if self.model == 'SHO':
            return SHO_energy(atoms)

        # Gaussian Process
        if self.model == 'GP':
            config = GP_config(self.atoms, self.cell)
            energy, sigma = GP_energy(config, self.ML_model)

            # # Check sigma bound
            # if la.norm(sigma) > .01:
            #     print("Warning: Uncertainty cutoff reached: sigma = {}".format(sigma))

            return energy

        # Kernel Ridge Regression
        if self.model == "KRR":
            # Helper function which converts atom positions into a configuration
            #  appropriate for the KRR model. Could then be modified
            #  later depending on alternate representations.
            config = KRR_configuration(self.atoms, self.cell)
            return KRR_energy(config, self.ML_model)

        # Lennard Jones
        if self.model == 'LJ':
            return LJ_energy(atoms)

    def update_atom_forces(self, atoms=None, cell=None, model=None, qe_config=None, fd_accuracy=None):
        """
        Perturbs the atoms by a small amount dx in each direction
        and returns the gradient (and thus the force)
        via a finite-difference approximation.

        If the engine's is in AIMD config,  runs quantum ESPRESSO on the current configuration and then
        stores the forces output from ESPRESSO as the atomic forces.
        """

        atoms = atoms or self.atoms
        cell = cell or self.cell
        model = model or self.model
        qe_config = qe_config or self.qe_config
        fd_accuracy = fd_accuracy or self.fd_accuracy

        if self.model == 'AIMD':
            print("AIMD model")
            results = qe_config.run_espresso(self.atoms, self.cell, qe_config=self.espresso_config)

            if self.verbosity == 4:
                print("Energy: {}".format(results['energy']))

            force_list = results['forces']
            for n in range(len(force_list)):
                self.atoms[n].force = list(np.array(results['forces'][n]) * 13.6 / 0.529177)

            return

        E0 = self.get_config_energy(self.atoms)

        if self.verbosity >= 4:
            print('\nEnergy: {}\n'.format(E0[0][0]))

        atoms = self.atoms

        # Finite-Difference Approx., 2nd/ 4th order as specified

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
                        print("Assigned force on atom's coordinate", coord, " to be ",
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

    def take_timestep(self, atoms=None, dt=None, method=None):
        """
        Propagate the atoms forward by timestep dt according to it's current force and previous position.
        Note that the first time step does not have the benefit of a previous position, so, we use
        a standard third-order Euler timestep incorporating information about the velocity and the force.
        """

        self.update_atom_forces()

        atoms = atoms or self.atoms
        # dt = dt  # or self.dt  # @STEVEN: COMMENTED THIS OUT -- MD_engine object has not attribute 'dt'
        method = method or "Verlet"
        self.time += dt

        temp_num = 0.

        # Third-order euler method, ref: https://en.wikipedia.org/wiki/Verlet_integration#Starting_the_iteration

        if method == 'TO_Euler':
            for atom in self.atoms:
                for coord in range(3):
                    atom.prev_pos[coord] = np.copy(atom.position[coord])
                    atom.position[coord] += atom.velocity[coord] * dt + atom.force[coord] * dt ** 2 / atom.mass
                    atom.velocity[coord] += atom.force[coord] * dt / atom.mass

        ################################################################################################################
        # Superior Verlet integration, ref: https://en.wikipedia.org/wiki/Verlet_integration
        ################################################################################################################
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
                    if self.verbosity == 5: print("Propagated a distance of ",
                                                  atom.position[coord] - atom.prev_pos[coord])

        if self.store_trajectory:
            for n in range(len(self.atoms)):
                self.trajs[n].append(np.copy(self.atoms[n].position))

        if self.assert_boundaries: self.assert_boundary_conditions()
        if self.verbosity > 3: self.print_positions()

    def run(self, tf, dt, ti=0):
        """
        Calculates the force and then advanced one timestep
        """

        n_step = 0

        if self.time == 0:

            if self.verbosity > 1:
                print("\n======================================\nStep: {}".format(n_step))
                print("Current time: {}\n".format(self.time))

            # Very first timestep often doesn't have the 'previous position' to use, instead use third-order Euler method
            # using information about the position, velocity (if provided) and force
            self.take_timestep(dt=dt, method='TO_Euler')

            # check uncertainty and retrain model if it exceeds specified threshold
            if (self.model == 'GP' or self.model == 'KRR') and self.uncertainty_threshold > 0:

                if self.gauge_model_uncertainty():
                    if self.verbosity == 5:
                        print("Uncertainty valid")

                # move to previous md step, compute DFT, update training set, retrain ML model
                else:
                    if self.verbosity == 5:
                        print("Uncertainty invalid, computing DFT\n")

                    self.qe_config.run_espresso(self.atoms, self.cell,
                                                iscorrection=True, stepcount=n_step)
                    self.take_timestep(dt=-dt)
                    self.retrain_ml_model()

            n_step += 1

        while self.time < tf:

            if self.verbosity > 1:
                print("\n======================================\nStep: {}".format(n_step))
                print("Current time: {}\n".format(self.time))

            self.take_timestep(dt=dt)

            # check uncertainty and retrain model if it exceeds specified threshold
            if (self.model == 'GP' or self.model == 'KRR') and self.uncertainty_threshold > 0:

                if self.gauge_model_uncertainty():
                    if self.verbosity == 5:
                        print("Uncertainty valid")
                    continue

                # move to previous md step, compute DFT, update training set, retrain ML model
                else:
                    if self.verbosity == 5:
                        print("Uncertainty invalid, computing DFT\n")

                    self.qe_config.run_espresso(self.atoms, self.cell,
                                                iscorrection=True, stepcount=n_step)
                    self.take_timestep(dt=-dt)
                    self.retrain_ml_model()

            n_step += 1

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

          And if we find that a, b, c not in [0,1)e modulo by 1.
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

        # print("T = ", self.time)
        for n in range(len(self.atoms)):

            pos = self.atoms[n].position
            force = self.atoms[n].force
            if forces:
                print('Atom %d:' % n, np.round(pos, decimals=4), ' Force:', np.round(force, decimals=6))
            else:
                print('Atom %d:' % n, np.round(pos, decimals=4))

    def gauge_model_uncertainty(self):
        """
        Ceck if prediction's uncertainty is within an acceptable bound given by threshold.
        If not, run QE, add to training set and retrain ML model

        :return False if uncertainty is higher than threshold, True otherwise
        """

        # Kernel Ridge Regression
        if self.model == "KRR":
            config = KRR_configuration(self.atoms, self.cell)
            uncertainty = KRR_uncertainty(config, self.ML_model)
            if self.verbosity >= 4:
                print("Uncertainty at current configuration:", uncertainty)
            if self.uncertainty_threshold < uncertainty:
                if self.verbosity > 2: print("!! The uncertainty of the model is outside of the bounds; calling DFT...")
                return False

        # Gaussian Process
        if self.model == "GP":

            config = GP_config(self.atoms, self.cell)
            energy, sigma = GP_energy(config, self.ML_model)

            if self.uncertainty_threshold < sigma:
                print(
                    "\nCAUTION: The uncertainty of the model is outside of the specified threshold: sigma = {}\n".format(
                        sigma[0]))
                return False

        return True

    def retrain_ml_model(self, model=None, ML_model=None):
        """"
        If uncertainty has surpassed cutoff value, update training set and retrain ML model
        """

        # model = model or self.model
        # ML_model = ML_model or self.ML_model

        # Kernel Ridge Regression
        if self.model == 'KRR':
            # TODO: MAKE KRR AND GP COHERENT CODE-WISE
            init_train_set = np.array(self.ML_model.total_train_set)
            init_train_ens = np.array(self.ML_model.total_train_ens)
            aug_train_set, aug_train_ens = get_aug_values(self.qe_config.correction_folder, ML_model=self.ML_model)
            positions = [aug_train_set[m * 3:(m + 1) * 3] for m in range(int(self.ML_model.vec_length / 3))]
            aug_train_set = KRR_configuration(atoms=None, cell=self.cell, point_set=positions)

            if self.verbosity == 5:
                print('Trying to reshape the augmented training set with shape ', aug_train_set.shape,
                      'into the shape of ', init_train_set.shape)
            new_train_set = np.concatenate((init_train_set, aug_train_set), axis=0)
            new_train_ens = np.append(init_train_ens, aug_train_ens)
            if self.verbosity == 5:
                print('Augmented and original set now has shape of:', new_train_set.shape)
            self.ML_model.fit(new_train_set, new_train_ens)
            self.ML_model.total_train_set = np.array(new_train_set)
            self.ML_model.total_train_ens = np.array(new_train_ens)

        # Gaussian Process
        if self.model == "GP":
            # update training set
            x_init = self.ML_model.original_train_set
            y_init = self.ML_model.original_train_ens

            x_add, y_add = get_aug_values(self.qe_config.correction_folder)

            x_upd = np.concatenate((x_init, np.asarray(x_add)[:, None].T), axis=0)
            y_upd = np.concatenate((y_init, np.asarray(y_add)[:, None]), axis=0)

            # retrain
            self.ML_model.fit(x_upd, y_upd)

            return


class Atom():
    """
    Class that holds basic information on atoms in the system

    # @STEVEN: PLESE CHECK WHETHER I SPECIFIED PARAMS CORRECTLY IN DOCS
    Parameters
    ----------
    position:       list, atom positions
    velocity:       list, atom velocities
    force:          list, forces on atoms
    initial_pos:    list, atom initial positions
    mass:           float, atom mass
    element:        str, chemical species
    constraint:     list of booleans, apply constraint or not
    fractional:     boolean, if coordinates are fractional in their original cell
    """
    mass_dict = {'H': 1.0, "Al": 26.981539, "Si": 28.0855}

    def __init__(self, position=[0., 0., 0.], velocity=[0., 0., 0.], force=[0., 0., 0.], initial_pos=[0, 0, 0],
                 mass=None,
                 element='', constraint=[False, False, False], fractional=False):

        self.position = np.array(position)

        # Used in Verlet integration
        self.prev_pos = np.array(self.position)
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.element = element

        self.fractional = bool(fractional)
        self.constraint = constraint

        self.fingerprint = rand.rand()  # This is how I tell atoms apart. Easier than indexing them manually...

        if self.element in self.mass_dict.keys() and mass == None:
            self.mass = self.mass_dict[self.element]
        else:
            self.mass = mass or 1.0

        # tests with SHO
        self.initial_pos = np.array(initial_pos)

        self.parameters = {'position': self.position,
                           'velocity': self.velocity,
                           'force': self.force,
                           'mass': self.mass,
                           'element': self.element,
                           'constraint': self.constraint,
                           'initial_pos': self.initial_pos,
                           'fractional coordinate:': self.fractional}

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
    def __init__(self, workdir=None, run_pwscf=None, pseudopotentials=None,
                 molecule=False, nk=15, correction_folder=None, system_name="AIMD", correction_number=0, ecut=40,
                 parallelization={'np': 1, 'nk': 0, 'nt': 0, 'nd': 0}):

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
        self.parallelization = dict(parallelization)

        self.system_name = system_name
        self.ecut = ecut
        self.nk = nk  # Dimensions of k point grid

        # Will be used for correction folders later
        self.correction_folder = correction_folder or self.workdir
        self.correction_number = self.get_correction_number()

    def get_correction_number(self):
        folders_in_correction_folder = list(os.walk(self.correction_folder))[0][1]

        steps = [fold for fold in folders_in_correction_folder if self.system_name + "_step_" in fold]

        if len(steps) >= 1:
            stepvals = [int(fold.split('_')[-1]) for fold in steps]
            correction_number = max(stepvals)
        else:
            return 0
        return correction_number + 1

    def run_espresso(self, atoms, cell, stepcount=0, iscorrection=False):
        pseudopots = {}
        elements = [atom.element for atom in atoms]
        for element in elements:
            pseudopots[element] = self.pseudopotentials[element]

        ase_struc = Atoms(symbols=[atom.element for atom in atoms],
                          positions=[atom.position for atom in atoms],
                          cell=cell,
                          pbc=[0, 0, 0] if self.molecule else [1, 1, 1])

        struc = Struc(ase2struc(ase_struc))

        if self.molecule:
            kpts = Kpoints(gridsize=[1, 1, 1], option='gamma', offset=False)
        else:
            nk = self.nk
            kpts = Kpoints(gridsize=[nk, nk, nk], option='automatic', offset=False)

        if iscorrection:
            dirname = 'step_' + str(self.system_name.title().strip('\'')) + '_' + str(stepcount)
        else:
            dirname = 'temprun'

        runpath = Dir(path=os.path.join(os.environ['PROJDIR'], "AIMD", dirname))

        input_params = PWscf_inparam({
            'CONTROL': {
                'prefix': self.system_name.title().strip('\''),
                'calculation': 'scf',
                'pseudo_dir': os.environ['ESPRESSO_PSEUDO'],
                'outdir': runpath.path,
                # 'wfcdir': runpath.path,
                'disk_io': 'low',
                'tprnfor': True,
                'wf_collect': False
            },
            'SYSTEM': {
                'ecutwfc': self.ecut,
                'ecutrho': self.ecut * 8,
                # 'nspin': 4 if 'rel' in potname else 1,

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
                                   parallelization=self.parallelization)
        output = parse_qe_pwscf_output(outfile=output_file)

        with open(runpath.path + '/en', 'w') as f:
            f.write(str(output['energy']))
        with open(runpath.path + '/pos', 'w')as f:
            for pos in [atom.position for atom in atoms]:
                f.write(str(pos) + '\n')

        return output


def first_derivative_2nd(fm, fp, h):
    """
    Computes the second-order accurate finite difference form of the first derivative
    which is (  fp/2 - fm/2)/(h) -- ref: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """
    if h == 0:
        print("Warning... Trying to divide by zero. Derivative will diverge.")
    return (fp - fm) / float(2 * h)


def first_derivative_4th(fmm, fm, fp, fpp, h):
    """
    Computes the fourth-order accurate finite difference form of the first derivative
    which is (fmm/12  - 2 fm /3 + 2 fp /3 - fpp /12)/h -- ref: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """
    if h == 0:
        print("Warning... Trying to divide by zero. Derivative will diverge.")

    return (fmm / 12. - 2 * fm / 3. + 2 * fp / 3. - fpp / 12.) / float(h)


def SHO_config(atoms):
    """
    No special input config needed for SHO
    """
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


def KRR_configuration(atoms=None, cell=None, point_set=None, fractional=True):
    """
    Pass the atoms either as a list of 'atoms', or as a list of lists each of length three representing
        the positions of the atoms.

    atoms: List of atoms
    cell: 3x3 matrix defining the cell of the system
    point_set: Alternate mode of input for debugging or for calling outside of the MD engine
    fractional: If true, will *convert* the positions into fractional coordinates. If the positions are *already*
            in fractional form, ensure that the cell is sized appropriately.
    """
    # Either pass atoms or make sure that positions is a list of lists of length three

    cell = None or cell

    # Fork between the two input methods
    if atoms != None:
        positions = [at.position for at in atoms]

        coords = np.empty(shape=(3, len(atoms)))
        if atoms[0].fractional is True:
            for n in range(len(atoms)):
                coords[:, n] = np.dot(cell, atoms[n].position)
            return coords.T.flatten().reshape(1, -1)

        else:
            return np.array(positions).flatten()

    else:
        coords = np.empty(shape=(3, int(len(point_set))))
        if fractional is True:
            for n in range(len(point_set)):
                coords[:, n] = np.dot(la.inv(cell), point_set[n])
            return coords.T.flatten().reshape(1, -1)
        else:
            return np.array(point_set).flatten().reshape(1, -1)


def KRR_energy(krrconfig, model):
    """
    Compute energy w/ Kernel Ridge Regression
    """
    return model.predict(krrconfig)


def GP_config(atoms, cell):
    """
    Compute input config for Gaussian Process
    :return: proper input config
    """
    coords = np.empty(shape=(3, len(atoms)))

    for n in range(len(atoms)):
        # TODO BY SIMON: CONVERT INV TO SOLVE LINEAR SYSTEM
        coords[:, n] = np.dot(la.inv(cell), atoms[n].position)

    coords = coords.flatten()
    coords = coords.reshape(1, -1)
    return coords


def KRR_uncertainty(krrconfig, model, kernel=False, measure='min', norm_order=2):
    """
    Computes a form of the uncertainty. Typically returns the euclidean norm between the
        input x and all of the x points in the training set. 'Kernel' flag uses the
        kernel as the thing to compute distances between. Measure is 'min' versus 'mean', which uses the
        minimum versus mean uncertainty.

    krrconfig: The KRR configuration
    model: The KRR model to compute the uncertainty of a configuration from
    measure: 'min' to return the minimum uncertainty, 'mean' to return the average uncertainty
    norm_order: Allows for different vector norm orders to be taken.
    """

    deviations = [la.norm(krrconfig - x, ord=norm_order) for x in model.total_train_set]

    if kernel == True:
        kerneldev = [model.gamma * dev ** 2 for dev in deviations]
    print(np.min(deviations))
    if measure == 'min':
        if kernel:
            return np.min(kerneldev)
        else:
            return np.min(deviations)
    if measure == 'mean':
        if kernel:
            return np.mean(kerneldev)
        else:
            return np.mean(deviations)


def GP_energy(gpconfig, model):
    """
    Compute energy w/ Gaussian Process
    """
    return model.predict(gpconfig, return_std=True)


def get_aug_values(correction_folder, keyword='step', ML_model=None):
    """
    Retrieves the position and energy information from a 'correction folder' which contains ESPRESSO runs
    used to train the dataset. Keyword denotes the substring in the name of a folder that is

    correction_folder: path to folder which contains folders of QE runs which will augment the training set
    keyword: substring to look for in a folder to parse it for the position and energy information

    """

    energies = []
    positions = []
    forces = []
    indices = []

    if len([folder for folder in list(os.walk(correction_folder))[0][1]]) > 0:
        # Iterate through the folders in the correction folder which contain the keyword
        for fold in [folder for folder in list(os.walk(correction_folder))[0][1] if keyword in folder]:
            # The folders should have a naming convention involving ending with an underscore and a number
            index = int(fold.split("_")[-1])
            if ML_model != None:
                if index not in ML_model.aug_indices:
                    ML_model.aug_indices.append(index)
                else:
                    continue

            fold = correction_folder + '/' + fold

            with open(fold + '/en', 'r') as f:
                energies.append(float(f.readlines()[0]))

            with open(fold + '/pos', 'r') as f:
                read_pos = f.readlines()
                for n in range(len(read_pos)):
                    curr_pos = read_pos[n].strip().strip('[').strip('\n').strip(']')
                    # print(curr_pos)
                    curr_pos = [float(x) for x in curr_pos.split()]
                    for x in curr_pos:
                        positions.append(x)

            # print("Found positions", positions, "and energies", energies)

            return positions, energies
    else:
        print("WARNING!! Correction folder", correction_folder, " seems to have no folders in it--",
              " attempt at sourcing files for augmentation runs failed! ")
        return None
