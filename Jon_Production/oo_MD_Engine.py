
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import numpy.random as rand
import numpy.linalg as la
from sklearn.kernel_ridge import KernelRidge


class Atom():
    def __init__(self, position=(0., 0., 0.), velocity=(0., 0., 0.), force=(0., 0., 0.), initial_pos=(0, 0, 0),
                 mass=None, element='', constraint=(False, False, False)):

        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.element = str(element)

        # Used in Verlet integration
        self.prev_pos = np.array(self.position)
        self.initial_pos = self.position if self.position!=(0,0,0) else initial_pos
        # Boolean which signals if the coordinates are fractional in their original cell
        self.constraint = list(constraint)
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
                           'initial_pos': self.initial_pos,
                           'fractional coordinate:': self.fractional}

    # Pint the
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








class Structure(list):
    """
    Class which stores list of atoms as well as information on the structure,
    which is acted upon by the MD engine.
    Parameterized by the structure_params object in the YAML input files.

    args:
    alat (float): Scaling factor for the lattice
    cell (3x3 nparray): Matrix of vectors which define the unit cell
    elements (list [str]): Names of elements in the system
    atom_list (list [Atom]): Atom objects which are propagated by the MD Engine
    """
    mass_dict = {'H': 1.0, "Al": 26.981539, "Si": 28.0855}

    def __init__(self, alat=1.,lattice=np.eye(3),elements=None, atom_list=None,fractional=True):

        self.atom_list = [] or atom_list
        self.elements = [] or set(elements)
        self.alat = alat

        if np.shape(lattice)!=(3,3):
            print("WARNING! Inappropriately shaped cell passed as input to structure!")
            raise Exception(np.shape(lattice))

        self.lattice = lattice
        self.fractional = fractional

        super(Structure, self).__init__(self.atom_list)


    def print_atoms(self,fractional=True):

        fractional = self.fractional

        if fractional:
            for n, at in enumerate(self.atom_list):
                print("{}:{} ({},{},{}) ".format(n, at.element, at.position[0], at.position[1], at.position[2]))
        else:
            for n, at in enumerate(self.atom_list):
                print("{}:{} ({},{},{}) ".format(n, at.element, at.position[0], at.position[1], at.position[2]))

    def __str__(self):
        self.print_atoms()

    def get_positions(self):
        return [at.position for at in self.atom_list]
    def set_forces(self,forces,):
        """
        Sets forces
        :param forces: List of length of atoms in system of length-3 force components
        :return:
        """
        if len(self.atom_list)!=len(forces):
            print("Warning! Length of list of forces to be set disagrees with number of atoms in the system!")
            Exception('Forces:',len(forces),'Atoms:',len(self.atom_list))
        for n,at in enumerate(self.atom_list):
            at.force = forces[n]

    def print_structure(self):
        cell = self.cell
        print('Alat:{}'.format(self.alat))
        print("Cell:\t [[ {}, {}, {}".format(cell[0,0],cell[0,1],cell[0,2]))
        print(" \t [ {},{},{}]".format(cell[1,0],cell[1,1],cell[1,2]))
        print(" \t [ {},{},{}]]".format(cell[2,0],cell[2,1],cell[2,2]))



def setup_structure(structure_config):

    sc = structure_config
    for n, at in enumerate(sc.positions):

    return Structure(alat=sc['alat'], lattice = sc['lattice'],elements=sc['elements'],)




class MD_Engine():
    def __init__(self, md_params, structure_config,qe_config,ml_config,hpc_config,
                  energy_or_force_driven='energy'):
        """
        Initialize the features of the system, which include:
        input_atoms: Atoms in the unit cell, list of objects of type Atom (defined later on in this notebook)
        cell:  3x3 matrix [[a11 a12 a13],..] which define the dimensions of the unit cell.
        dx:    Perturbation distance used to evaluate the forces by finite differences in potential energy.
        verbosity: integer from 0 to 5 which determines how much information will be printed about runs.
                    Ranging from 0 as silent to 5 is TMI and is mostly for debugging.
        model: The energy model used.
        store_trajectory: Boolean which determines if positions will be saved after every time step.
        qe_config: Espresso_config object which determines how on-the-fly quantum espresso runs will be
                        parameteried.
        thermo_config:   Thermo config object which determines the thermostate methodology that will be used
                        for molecular dynamics.
        ML_model: Object which parameterizes a ML model which returns energies when certain configurations are input.
        assert_boundaries : Determines if boundary conditions are asserted at the end of each position update
                            (which forces atoms to lie within the unit cell).
        fd_accuracy: Integer 2 or 4. Determines if the finite difference force calculation will be evaluated to
                        second or fourth order in dx. Note that this increases the number of energy calculations
                        and thus increases run time.
        uncertainty_threshold: Defines an uncertainty cutoff above which a DFT run will be called.
        energy_or_force_driven: Maps to 'driver': dictates if the finite difference energy model drives the simulation
                                or if the force comes directly from the model (such as with AIMD.)
        """
        self.verbosity = md_config.get('verbosity',1)

        self.atoms = input_atoms  # Instantiate internal list of atoms, populate via helper function
        for atom in input_atoms:
            if self.verbosity == 5:
                print('Loading in', atom)

        self.dx = md_config.get('fd_dx',.1)
        self.cell = structure_config.get('unit_cell')
        self.model = model

        self.time = 0.

        self.store_trajectory = md_config.get('store_trajectory',True)

        # Construct trajectories object
        if store_trajectory:
            self.trajs = []
            for n in range(len(self.atoms)):
                self.trajs.append([])

        # Set configurations
        self.qe_config = qe_config
        self.ml_config = ml_config

        self.assert_boundaries = md_params.get('assert_boundaries',False)

        self.fd_accuracy = None or fd_accuracy

        self.uncertainty_threshold = None or uncertainty_threshold
        self.energy_or_force_driven = energy_or_force_driven

    def get_config_energy(self, atoms=None):
        """
        The positions of the atoms are passed to a given model of choice,
        in which the positions are converted into the configuration which is
        appropriate to be parsed by the model.

        For instance, the positions may need to be converted into
        atom-centered symmetry functions for the Gaussian Process model. By
        passing in the positions, the necessary mapping is performed, and then
        the appropriate model is called.
        """

        atoms = atoms or self.atoms

        # Simple Harmonic Oscillator
        if self.model == 'SHO':
            return SHO_energy(atoms)

        #######
        # Below are 'hooks' where we will later plug in to our machine learning models
        #######
        if self.model == 'GP':
            config = GP_config(positions)
            energy, uncertainty = GP_energ(config, self.ML_model)
            if np.norm(uncertainty) > .01:
                print("Warning: Uncertainty cutoff found.")

            return GP_energy(config, self.ML_model)

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

        Or, if the engine's model is ab-initio molecular dynamics,
        runs Quantum ESPRESSO on the current configuration and then
        stores the forces output from ESPRESSO as the atomic forces.
        """
        atoms = atoms or self.atoms
        cell = cell or self.cell
        model = model or self.model
        qe_config = qe_config or self.qe_config
        fd_accuracy = fd_accuracy or self.fd_accuracy

        if self.energy_or_force_driven == "energy" and self.model == "AIMD":
            print("WARNING! You are asking for an energy driven model with Ab-Initio MD;",
                  " AIMD is configured only to work on a force-driven basis.")

        if self.model == 'AIMD':
            results = qe_config.run_espresso(atoms, cell, qe_config)

            if self.verbosity == 4: print("E0:", results['energy'])

            force_list = results['forces']
            for n in range(len(force_list)):
                atoms[n].force = list(np.array(results['forces'][n]) * 13.6 / 0.529177)

            return

        if self.energy_or_force_driven == "force":
            # stuff_you_need_to_get
            # force_list = placeholder_force_function()
            pass

        E0 = self.get_config_energy(atoms)
        if self.verbosity == 4: print('E0:', E0)

        # Depending on the finite difference accuracy specified at instantiation,
        # perform either 2 or 4 potential energy calculations.

        # Second-order finite-difference accuracy
        if fd_accuracy == 2:
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
        if fd_accuracy == 4:
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

        for atom in atoms:
            atom.apply_constraint()

    def take_timestep(self, atoms=None, dt=None, method=None):
        """
        Propagate the atoms forward by timestep dt according to it's current force and previous position.
        Note that the first time step does not have the benefit of a previous position, so, we use
        a standard third-order Euler timestep incorporating information about the velocity and the force.
        """

        self.update_atom_forces()

        atoms = atoms or self.atoms
        dt = dt or self.dt
        method = method or "Verlet"
        self.time += dt

        temp_num = 0.

        # Third-order euler method
        # Is a suggested way to begin a Verlet run, see:
        # https://en.wikipedia.org/wiki/Verlet_integration#Starting_the_iteration

        if method == 'TO_Euler':
            for atom in atoms:
                for coord in range(3):
                    atom.prev_pos[coord] = np.copy(atom.position[coord])
                    atom.position[coord] += atom.velocity[coord] * dt + atom.force[coord] * dt ** 2 / atom.mass
                    atom.velocity[coord] += atom.force[coord] * dt / atom.mass

        ######
        # Superior Verlet integration
        # Citation:  https://en.wikipedia.org/wiki/Verlet_integration
        ######

        elif method == 'Verlet':
            for atom in atoms:
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
            for n in range(len(atoms)):
                self.trajs[n].append(np.copy(atoms[n].position))

        if self.assert_boundaries: self.assert_boundary_conditions()
        if self.verbosity > 3: self.print_positions()

    def run(self, tf, dt, ti=0):
        """
        Handles timestepping; at each step, calculates the force and then
        advances via the take_timestep method.
        """

        # The very first timestep often doesn't have the 'previous position' to use,
        # so the third-order Euler method starts us off using already-present
        # information about the position, velocity (if provided) and force:

        if self.time == 0:
            self.take_timestep(dt=dt, method='TO_Euler')

        # Primary iteration loop
        # Most details are handled in the timestep function
        while self.time < tf:

            self.take_timestep(dt=dt)

            if (self.model == 'GP' or self.model == 'KRR') and self.uncertainty_threshold > 0:

                if self.gauge_model_uncertainty():
                    continue
                else:
                    # print("Timestep with unacceptable uncertainty detected! \n Rewinding one step, and calling espresso to re-train the model.")
                    self.qe_config.run_espresso(self.atoms, self.cell,
                                                iscorrection=True)
                    self.retrain_ml_model(self.model, self.ML_model)
                    self.take_timestep(dt=-dt)

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
        """
        Prints out the current positions of the atoms line-by-line,
        and also the forces
        """
        print("T=", self.time)
        for n in range(len(self.atoms)):

            pos = self.atoms[n].position
            force = self.atoms[n].force
            if forces:
                print('Atom %d:' % n, np.round(pos, decimals=4), ' Force:', np.round(force, decimals=6))
            else:
                print('Atom %d:' % n, np.round(pos, decimals=4))

    def gauge_model_uncertainty(self):
        """
        For later implementation with the Gaussian Process model.
        Will check to see if the uncertainty of the model's prediction of the energy
        within a given configuration is within an acceptable bound given by threshold.

        If it is unacceptable, the current configuration is exported into a Quantum ESPRESSO run,
        which is then added into the ML dataset and the model is re-trained.
        """
        if self.model == "KRR":

            config = KRR_configuration(self.atoms, self.cell)
            uncertainty = KRR_uncertainty(config, self.ML_model)
            if self.verbosity >= 4:
                print("Uncertainty at current configuration:", uncertainty)
            if self.uncertainty_threshold < uncertainty:
                if self.verbosity > 2: print("!! The uncertainty of the model is outside of the bounds; calling DFT...")
                return False

        if self.model == 'GP':
            config = GP_config(positions)
            energy, uncertainty = GP_energ(config, self.ML_model)
            if np.norm(uncertainty) > self.uncertainty_threshold:
                print("Warning: Uncertainty cutoff met.")

            return GP_energy(config, self.ML_model)

            pass

        return True

    def retrain_ml_model(self, model=None, ML_model=None):
        model = model or self.model
        ML_model = ML_model or self.ML_model

        if self.model == 'KRR':

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
            # print('The size of the total train set is now',self.ML_model.total_train_set.shape)
            # print("The size of the total energy set is now",self.ML_model.total_train_ens)
            return

        if self.model == "GP":
            pass
