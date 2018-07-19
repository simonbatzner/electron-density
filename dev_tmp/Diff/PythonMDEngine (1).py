# coding: utf-8

# # In this notebook, we define a simple Python-based Molecular Dynamics engine which will allow for the use of ML-based Ab-Initio Molecular Dynamics Analogues.

# In[1]:


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import numpy.random as rand
import numpy.linalg as la
from sklearn.kernel_ridge import KernelRidge

# ## Change this according to your local environment to point to the 'util' folder

# In[2]:


sys.path.append(os.environ['PROJDIR'] + '/util')
sys.path.append('/Users/steven/Documents/Research/Projects/MLED/ML-electron-density/util')


# First, we define the engine which will power the whole simulation.
# 
# The engine object needs to keep track of every atom's individual position, and will propagate the system forward by computing the forces acting on each atom and then propagating forward by a time step.

# # Define the Atom Class, which is integral to the MD simulation (the main object of the engine)

# In[3]:


class Atom():
    mass_dict = {'H': 1.0, "Al": 26.981539, "Si": 28.0855}

    def __init__(self, position=(0., 0., 0.), velocity=(0., 0., 0.), force=(0., 0., 0.), initial_pos=(0, 0, 0),
                 mass=None,
                 element='', constraint=(False, False, False), fractional=False):

        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.element = str(element)

        # Used in Verlet integration
        self.prev_pos = np.array(self.position)

        # Boolean which signals if the coordinates are fractional in their original cell
        self.fractional = bool(fractional)
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


# TODO: Function to turn an ASE structure into a list of atoms
def ase_to_atoms(structure):
    pass


# # Define the MD Engine, which contains functions for:
# - Getting the energy of the structural configuration
# - Updating the forces acting on the atoms within the system
# - Advancing the system forward by a timestep
# 
# 
# Note that even when passing variables which are members of the MD_Engine class, we will pass the objects themselves (except for verbosity) so as to make transparent what goes into each function, as well as to enable easier debugging.
# 
# ### One begins a run of the MD engine by calling the run method. The run method works as follows:
# -  The first timestep is taken as a third-order accurate Euler Method. If information about the velocity is provided this is extra accurate. The successive timesteps are by default taken in the fourth-order accurate Verlet method.
# -  The take timestep method is called in a while loop.
# -  There are different ways for it to run:
#     * Finite-difference-energy involves perturbing every atom by dx in each of the three cartesian directions and gauging the change in energy of the system as a whole. 
#     * Another involves getting the forces directly from an ML model, or by ESPRESSO runs (ab-initio Molecular Dynamics).
# - After forces are computed, then the system steps forward in time via the Verlet integration algorithm.
# -  At the conclusion of each timestep, if the uncertainty threshold is nonzero and molecular dynamics are being driven by a ML model, then the uncertainty is gauged. If it exceeds the threshold a DFT run is performed and the model is re-trained.
# - This repeats until conclusion.

# In[4]:


class MD_engine():
    def __init__(self, input_atoms=[], cell=np.eye(3), dx=.1, verbosity=1, model='SHO', store_trajectory=True,
                 qe_config=None, thermo_config=None, ML_model=None, assert_boundaries=True, fd_accuracy=2,
                 uncertainty_threshold=0, energy_or_force_driven='energy'):
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
        self.qe_config = None or qe_config
        self.thermo_config = None or thermo_config
        self.ML_model = None or ML_model

        self.assert_boundaries = None or assert_boundaries

        self.fd_accuracy = None or fd_accuracy

        self.uncertainty_threshold = None or uncertainty_threshold
        self.driver = energy_or_force_driven

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

        if self.model == 'AIMD':
            results = qe_config.run_espresso(atoms, cell, qe_config)

            if self.verbosity == 4: print("E0:", results['energy'])

            force_list = results['forces']
            for n in range(len(force_list)):
                atoms[n].force = list(np.array(results['forces'][n]) * 13.6 / 0.529177)

            return

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


# # Set up KRR, Energy, Uncertainty

# In[5]:


# todo: Roll all of these into an augmented "KRR" class which contains all of the associated methods and such as
#      wrappers around SKlearn.

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
    Return the energy of a given configuration krrconfig associated with a model 
    """
    return model.predict(krrconfig.reshape(1, -1))


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
    else:
        print("WARNING!! Correction folder", correction_folder, " seems to have no folders in it--",
              " attempt at sourcing files for augmentation runs failed! ")
        return None


print("Brief unit test that the KRR config function is working roughly right:")
Hatom1 = Atom(position=[.24, 0, 0], element='H')
Hatom2 = Atom(position=[1.75, 0, 0], element='H')
Hatom3 = Atom(position=[1, 2, 1], element='H')
atoms = [Hatom1, Hatom2, Hatom3]
cell = [[4.10, 0, 0], [0, 4.10, 0], [0, 0, 4.10]]
try:
    assert (np.array_equal(KRR_configuration(atoms, cell), np.array([0.24, 0.0, 0.0, 1.75, 0.0, 0.0, 1.0, 2.0, 1.0])))
    print("Passed.")
except:
    pass

# ### Quantum Cafe: Brew ESPRESSO

# In the ESPRESSO config object, tune the parameters according to your system. You could use environment variables like so (defined in your home directory's .bash_profile) or just hard-code them into the notebook below.
# 
# This is going to be important for letting the Gaussian Process model interface with ESPRESSO, in that positions will be printed corresponding to the 

# In[6]:


workdir = os.environ['PROJDIR'] + '/AIMD'

# In[7]:


from project_pwscf import *
from project_objects import *
from ase import Atoms
import os


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

    def run_espresso(self, atoms, cell, iscorrection=False):

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
            self.correction_number = self.get_correction_number()
            # print("rolling with correction number",qe_config.correction_number)
            dirname = self.system_name + '_step_' + str(self.correction_number)
        else:
            dirname = 'temprun'
        runpath = Dir(path=os.path.join(os.environ['PROJDIR'], "AIMD", dirname))
        input_params = PWscf_inparam({
            'CONTROL': {
                'prefix': self.system_name,
                'calculation': 'scf',
                'pseudo_dir': os.environ['ESPRESSO_PSEUDO'],
                'outdir': runpath.path,
                #            'wfcdir': runpath.path,
                'disk_io': 'low',
                'tprnfor': True,
                'wf_collect': False
            },
            'SYSTEM': {
                'ecutwfc': self.ecut,
                'ecutrho': self.ecut * 8,
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
                                   ncpu=1, parallelization=self.parallelization)
        output = parse_qe_pwscf_output(outfile=output_file)

        with open(runpath.path + '/en', 'w') as f:
            f.write(str(output['energy']))
        with open(runpath.path + '/pos', 'w')as f:
            for pos in [atom.position for atom in atoms]:
                f.write(str(pos) + '\n')

        return output


conf_mol = ESPRESSO_config(molecule=True, ecut=90, nk=1)
conf_solid = ESPRESSO_config(molecule=False, ecut=40, nk=10)
conf_solid_test = ESPRESSO_config(molecule=False, ecut=10, nk=5, system_name='Al')

# Unit Test for ESPRESSO

# In[8]:


import os

Hatom3 = Atom(position=[0, 0, 0], element='H')
Hatom4 = Atom(position=[.5, 0, 0], element='H')

atoms2 = [Hatom3, Hatom4]

print("Unit test that DFT is being called correctly...")
results = conf_mol.run_espresso(atoms=atoms2, cell=[[2, 0, 0], [0, 5, 0], [0, 0, 1.5]])
assert (np.abs(results['energy'] + 40.55) < 0.1)
print("Passed.")

# ## KRR Uncertainty Testing Grounds

# In[9]:


import sys

sys.path.append("/Users/steven/Documents/Research/Projects/MLED/ML-electron-density/Solid_State")
import os
import numpy as np
import scipy as sp
from KRR_reproduce import *
from KRR_Functions import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)
kcal_conv = 0.043

# # Testing Ground

# In[10]:


Hatom1 = Atom(position=[.24, 0, 0], element='H')
Hatom2 = Atom(position=[.75, 0, 0], element='H')

atoms = [Hatom1, Hatom2]
eng = MD_engine(cell=[[5, 0, 0], [0, 1, 0], [0, 0, 1]], input_atoms=atoms, verbosity=4, model='AIMD', dx=.001,
                qe_config=conf_mol)
eng.run(.2, .1)

# # Concisely Load in Aluminum

# In[11]:


STR_PREF = '/Users/steven/Documents/Research/Projects/MLED/ML-electron-density/data/Aluminum_Dataset/store/'
pos = [];
ens = []
for n in range(201):
    ens.append(np.reshape(np.load(STR_PREF + 'en_store/energy' + str(n) + '.npy'), (1))[0])
    pos.append(np.load(STR_PREF + 'pos_store/pos' + str(n) + '.npy').reshape(-1, 1).flatten())

ens = np.array(ens);
pos = np.array(pos)
input_dim = 12

# choose training set sizes by commenting
M = 15
alpha = 1.5848931924611107e-05
gamma = 0.14174741629268056

M = 200
alpha = 2.511886431509572e-09
gamma = 0.06579332246575682

AlKRR = fit_quick(pos, ens, alpha=alpha, gamma=gamma)
AlKRR.vec_length = 12
AlKRR.original_train_set = np.array(pos)
AlKRR.original_train_ens = np.array(ens)
AlKRR.gamma = gamma
AlKRR.alpha = alpha
AlKRR.total_train_set = np.array(pos)
AlKRR.total_train_ens = np.array(ens)
AlKRR.aug_indices = []

test_pos = np.array(pos[50])
print("Testing to see if energy of config 50 of total data set has correct energy...")
try:
    assert (np.abs(AlKRR.predict(test_pos.reshape(1, -1))[0] + 227.8) < .1)
    print("Passed.")
except:
    assert (False)

print(KRR_uncertainty(test_pos, AlKRR, kernel=True))
test_pos[0] += 1
print(KRR_uncertainty(test_pos, AlKRR, kernel=True))

# In[12]:


print(pos[1])
alat = 4.10
Npos = 15
Atom1 = Atom(position=pos[Npos][:3] * alat + .2, element='Al', fractional=True)
Atom2 = Atom(position=pos[Npos][3:6] * alat, element='Al', fractional=True)
Atom3 = Atom(position=pos[Npos][6:9] * alat, element='Al', fractional=True)
Atom4 = Atom(position=pos[Npos][9:] * alat, element='Al', fractional=True)
atoms = [Atom1, Atom2, Atom3, Atom4]

conf_solid_test.system_name = 'Al'
conf_solid_test.correction_folder = os.environ['PROJDIR'] + '/AIMD'
conf_solid_test.ecut = 10
conf_solid_test.nk = 3
# Set to .07 threshold if you want it to fire
KRR_eng = MD_engine(cell=alat * np.eye(3), input_atoms=atoms, ML_model=AlKRR, model='KRR',
                    store_trajectory=True, verbosity=4, assert_boundaries=False, dx=.01, fd_accuracy=4,
                    uncertainty_threshold=.07, qe_config=conf_solid_test)

KRR_eng.run(.5, .1)

# # Silicon Test
# (Some code borrowed from Jon Vandermause)

# # Testing KRR as an engine for MD

# In[13]:


print(pos[1])
alat = 4.10
Npos = 15
Atom1 = Atom(position=pos[Npos][:3] * alat, element='Al', fractional=True)
Atom2 = Atom(position=pos[Npos][3:6] * alat, element='Al')
Atom3 = Atom(position=pos[Npos][6:9] * alat, element='Al')
Atom4 = Atom(position=pos[Npos][9:] * alat, element='Al')
atoms = [Atom1, Atom2, Atom3, Atom4]

KRR_eng = MD_engine(cell=alat * np.eye(3), input_atoms=atoms, ML_model=AlKRR, model='KRR',
                    store_trajectory=True, verbosity=4, assert_boundaries=False, dx=.1, fd_accuracy=4,
                    uncertainty_threshold=0)

KRR_eng.run(1, .1)

# In[14]:


alat = 4.10

Atom1 = Atom(position=pos[Npos][:3] * alat, element='Al')
Atom2 = Atom(position=pos[Npos][3:6] * alat, element='Al')
Atom3 = Atom(position=pos[Npos][6:9] * alat, element='Al')
Atom4 = Atom(position=pos[Npos][9:] * alat, element='Al')
atoms = [Atom1, Atom2, Atom3, Atom4]
config2 = ESPRESSO_config()
config2.molecule = False
Al_AIMD_eng = MD_engine(cell=alat * np.eye(3), input_atoms=atoms, model='AIMD',
                        qe_config=config2, store_trajectory=True, verbosity=4, assert_boundaries=False)

# # Jon's Silicon Run

# In[23]:


from ase.spacegroup import crystal
from ase.build import *

md_file = File({'path': '/Users/steven/Documents/Research/Projects/MLED/ML-electron-density/Si_Supercell_MD/si.md.out'})
md_run = parse_qe_pwscf_md_output(md_file)

silicon_cell = np.array([[0.00000, 8.14650, 8.14650],
                         [8.14650, 0.00000, 8.14650], [8.14650, 8.14650, 0.00000]])
alat = 5.431  # lattice parameter of si in angstrom


def make_struc_super(alat, dim):
    unitcell = crystal('Si', [(0, 0, 0)], spacegroup=227, cellpar=[alat, alat, alat, 90, 90, 90], primitive_cell=True)
    multiplier = np.identity(3) * dim
    ase_supercell = make_supercell(unitcell, multiplier)
    structure = Struc(ase2struc(ase_supercell))
    return structure


# perturb the positions of a supercell
def perturb_struc(alat, dim, pert_size):
    struc_pert = make_struc_super(alat, dim)
    for n in range(len(struc_pert.content['positions'])):
        for m in range(3):
            # get current coordinate
            coord_curr = struc_pert.content['positions'][n][1][m]

            # get perturbation by drawing from uniform
            pert = np.random.uniform(-pert_size, pert_size)

            # perturb the coordinate
            struc_pert.content['positions'][n][1][m] += pert

    return struc_pert


# pert_size=.1
Si_super = make_struc_super(alat, 3)
# print(Si_super)
positions = [pos[1] for pos in Si_super.positions]

Si_atoms = [Atom(position=positions[n], element='Si') for n in range(len(positions))]

print('Loaded in %d atoms' % len(Si_atoms))
# print(md_run.keys())
set_pos = np.array([np.array(md_run[n]['positions']).flatten() for n in range(1, 1001)])
set_ens = np.array([md_run[n]['energy'] for n in range(1, 1001)])
set_forces = np.array([md_run[n]['forces'] for n in range(1, 1001)])
# print(train_pos)
# print(train_ens)
# Si_kr = KernelRidge(kernel='rbf',alpha = alpha, gamma = gamma)

train_pos = [set_pos[n] for n in np.arange(0, len(set_pos), 2)]
test_pos = [set_pos[n] for n in np.arange(1, len(set_pos), 2)]

train_ens = [set_ens[n] for n in np.arange(0, len(set_pos), 2)]
test_ens = [set_ens[n] for n in np.arange(1, len(set_pos), 2)]

train_pos = [set_pos[n] for n in np.arange(0, 1000, 1)]
test_pos = [set_pos[n] for n in np.arange(0, 1000, 1)]

train_ens = [set_ens[n] for n in np.arange(0, 1000, 1)]
test_ens = [set_ens[n] for n in np.arange(0, 1000, 1)]
# print(train_pos[5][5*3:5*3+3])
# print(Si_atoms[5].position)

SiKRR = fit_quick(train_pos, train_ens, alpha=alpha, gamma=gamma)
# print(train_pos[0])
# print([at.position for at in Si_atoms])
# print(np.array([at.position for at in Si_atoms]).flatten() - np.array(train_pos[0]))

# config = KRR_config(Si_atoms,silicon_cell)

# print(train_pos[0])
# print(config)
# print(config - np.array(train_pos[0]))

# [kr, y_kr, errs, MAE, max_err] = fit_KS(train_pos, train_ens, \
#                                            test_pos, test_ens, alphas, \
#                                            gammas, cv=None)

# print(kr.best_estimator_)

# print( KRR_energy(config,Si_kr))
# print(Si_kr)





# ## Si Hyperparameter Optimization

# In[24]:


import warnings

warnings.filterwarnings("ignore")

alphas = np.logspace(-20, -3, 2)
gammas = np.logspace(-6, 1, 2)

alphas = np.append(alphas, [0.00020235896477251638, 0])
print(alphas)
cv = 9
gammas = np.append(gammas, 0.0009102981779915227)
[kr, y_kr, errs, MAE, max_err] = fit_KS(train_pos, train_ens, test_pos, test_ens, alphas, gammas, cv)
print(kr.best_estimator_)

# plot predictions
plt.figure()
plt.plot(y_kr, 'x', label='KS ' + str(M))
plt.plot(test_ens, label='DFT energy')
plt.xlabel('test point')
plt.ylabel('total energy (eV)')
plt.show()

# plot errors
plt.figure()
plt.plot(y_kr - test_ens, label='KS ' + str(M))
plt.axhline(0)
plt.xlabel('test point')
plt.ylabel('total energy (eV)')
plt.legend()
plt.show()

print('The KRR max error in kcal/mol is ' + str(max_err / kcal_conv))
print('The KRR MAE in kcal/mol is ' + str(MAE / kcal_conv))

# In[25]:



train_pos = [set_pos[n] for n in np.arange(0, 1000)]
train_ens = [set_ens[n] for n in np.arange(0, 1000)]

SiKRR = fit_quick(train_pos, train_ens, alpha=0.00020235896477251638, gamma=0.0009102981779915227)

positions = [pos[1] for pos in Si_super.positions]
Si_atoms = [Atom(position=positions[n], element='Si') for n in range(len(positions))]

Si_eng = MD_engine(cell=silicon_cell, input_atoms=Si_atoms, model='KRR',
                   ML_model=SiKRR, store_trajectory=True, verbosity=3, assert_boundaries=False, dx=.005, fd_accuracy=4)


#
# Si_eng.run(1,.1)



# 
# 
# # Unit Tests Below
# 
# 

# ## Set up SHO and LJ configuration + Energy functions
# 
# Note that this SHO energy is a very simple model which only has the atoms oscillating about their initial positions; there is no interactions between the atoms whatsoever. This is merely built as a test of the MD engine to ensure that it is running and propagating the atoms correctly.

# In[26]:


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


Hatom1 = Atom(position=[.26, 0, 0], initial_pos=[.25, 0, 0], element='H')
Hatom2 = Atom(position=[.75, 0, 0], element='H')
atoms = [Hatom1, Hatom2]
eng = MD_engine(cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], input_atoms=atoms, verbosity=0, model='LJ', dx=.001)
eng.run(.5, .001)

particle1traj = [x for x in eng.trajs[0]]
print("Unit testing if the LJ potential worked as predicted:")
assert (np.abs(particle1traj[5][0] - 0.25927) < .1)
print("Passed.")

# # TODO:
# - Rewrite to use ASE structures instead of 'atoms' and cells
# - Augmented training set checks for duplicates
# 

# In[27]:


ncpu = 1
parallelization = {'nk': 5}
parallelization_str = ""
if parallelization != {}:
    for key, value in parallelization.items():
        if value != 0:
            parallelization_str += '-%s %d ' % (key, value)
else:
    parallelization_str = "-np %d" % (ncpu)
pwscf_command = "mpirun {} {} < {} > {}".format(parallelization_str, 'hi', 'yo', 'out')
print(pwscf_command)

# In[ ]:



p
