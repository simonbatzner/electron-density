#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name, too-many-arguments

""""

Steven Torrisi, simon Batzner
"""

import sys
import os
import pprint
import time as time
import yaml

import numpy as np
import numpy.random as random
import numpy.linalg as la

from util.util import prepare_dir
from Jon_Production.utility import write_file, run_command


# TODO: Move this to utility file once we're done with everything else
def flatten_dict(d):
    """
    Recursively flattens dictionary
    :param d: dict to flatten
    :return: flattened dict
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


# TODO: Move this to utility file once we're done with everything else here
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# ------------------------------------------------------
#              ML Config
# ------------------------------------------------------


class ml_config(dict):
    """
    Creates an ml_config object.
    Args:
        params (dict): A set of input parameters as a dictionary.
    """

    def __init__(self, params, print_warn=True):

        super(ml_config, self).__init__()

        # init with default
        if params:
            self.update(params)

        # check if any parameters are missing
        missing = []

        # default parameters for machine learning model
        default_ml = {'regression_model': 'GP',
                      'gp_params': {'length_scale': 1,
                                    'length_scale_min': 1e-5,
                                    'length_scale_max': 1e5,
                                    'threshold_params': {'force_conv': 25.71104309541616,
                                                         'thresh_perc': .2}},
                      'fingerprint_params': {'eta_lower': 0,
                                             'eta_upper': 2,
                                             'eta_length': 10,
                                             'cutoff': 8}
                      }

        flat = flatten_dict(params)

        for key in flatten_dict(default_ml).keys():

            # use random to avoid .get() returning False for input values specified as 0
            rand = np.random.rand()

            if flat.get(key, rand) == rand:
                missing.append(key)

        if print_warn and missing != []:
            print("WARNING! Missing ML parameter, running model with default for: {}".format(missing))


# ------------------------------------------------------
#              MD Config
# ------------------------------------------------------

class MD_Config(dotdict):
    """
    Creates an MD_Config object.
    Base class of MD Engine.
    Args:
        params (dict): A set of input parameters as a dictionary.
    """

    def __init__(self, params, warn=True):

        self._params = ['name', 'comment', 'mode', 'ti', 'frames', 'mass',
                        'fd_dx', 'verbosity', 'assert_boundaries', 'timestep_method',
                        'fd_accuracy', 'energy_or_force']

        super(MD_Config, self).__init__(params)

        if params:
            self.update(params)

        default_md = {'name': time.strftime('%y.%m.%d-%H:%M:%S'),
                      'comment': "",
                      "mode": 'ML',
                      'ti': 0,
                      'frames': 100,
                      'dt': .1,
                      'mass': 0,
                      'fd_dx': .01,
                      'verbosity': 1,
                      'assert_boundaries': False,
                      'timestep_method': 'Verlet',
                      'fd_accuracy': 4,
                      'energy_or_force': 'Force'}

        # If value was not specified, load in default values.

        if self.get('frames', False) and self.get('tf', False) and self.get('dt', False) and warn:
            print("WARNING! Frames and final time specified;"
                  " these two parameters cannot both exist. Pick either a number of timesteps"
                  " or an automatically computed computer of timesteps.")
            raise Exception("Number of frames and final time both specified.")

        if self.get('frames', False) and self.get('tf', False) and self.get('dt', False):
            print("WARNING! Frames and final time specified;"
                  " these two parameters cannot both exist. Pick either a number of timesteps"
                  " or an automatically computed computer of timesteps.")
            raise Exception("Number of frames and final time both specified.")

        self.time = 0 + self.get('ti', 0)
        self.frame = 0
        missing = []

        for key in default_md.keys():

            if key not in self.keys():
                missing.append(key)

        for key in missing:
            print("Achtung! Missing MD parameter, running model with default for: {}".format(missing))
            self[key] = default_md[key]


# ------------------------------------------------------
#              Structure Objects
# ------------------------------------------------------


class Atom(object):
    def __init__(self, position=(0., 0., 0.), velocity=(0., 0., 0.), force=(0., 0., 0.), initial_pos=(0, 0, 0),
                 mass=None, element='', constraint=(False, False, False)):

        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.element = str(element)
        self.trajectory = {}  # keys: time rounded to 3 decimal places, values: {'position':[array], 'force':[array]}

        # Used in Verlet integration
        self.prev_pos = np.array(self.position)
        self.initial_pos = self.position if self.position.all != (0, 0, 0) else initial_pos
        # Boolean which signals if the coordinates are fractional in their original cell
        self.constraint = list(constraint)
        self.fingerprint = random.rand()  # This is how I tell atoms apart. Easier than indexing them manually...

        self.mass = mass or 1.0

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

    def __init__(self, atoms, alat=1., lattice=np.eye(3), fractional=True):

        self.atoms = [] or atoms
        self.elements = [at.element for at in atoms]
        self.species = [] or set(self.elements)
        self.alat = alat
        self.lattice = lattice
        self.fractional = fractional

        self.trajectory = {}

        super(Structure, self).__init__(self.atoms)

    def get_pprint_atoms(self):

        fractional = self.fractional
        report = ''
        if fractional:
            for n, at in enumerate(self.atoms):
                report += "{}:{} ({},{},{}) \n".format(n, at.element, at.position[0], at.position[1], at.position[2])
        else:
            for n, at in enumerate(self.atoms):
                report += "{}:{} ({},{},{}) \n".format(n, at.element, at.position[0], at.position[1], at.position[2])

        return report

    def __str__(self):
        return self.get_pprint_atoms()

    def get_positions(self):
        return [atom.position for atom in self.atoms]

    def get_positions_and_element(self):
        return [[atom.element, atom.position] for atom in self.atoms]

    def set_forces(self, forces):
        """
        Sets forces
        :param forces: List of length of atoms in system of length-3 force components
        :return:
        """
        if len(self.atoms) != len(forces):
            print("Warning! Length of list of forces to be set disagrees with number of atoms in the system!")
            Exception('Forces:', len(forces), 'Atoms:', len(self.atoms))
        for n, at in enumerate(self.atoms):
            at.force = forces[n]

    def print_structure(self):
        lattice = self.lattice
        print('Alat:{}'.format(self.alat))
        print("Cell:\t [[ {}, {}, {}".format(lattice[0, 0], lattice[0, 1], lattice[0, 2]))
        print(" \t [ {},{},{}]".format(lattice[1, 0], lattice[1, 1], lattice[1, 2]))
        print(" \t [ {},{},{}]]".format(lattice[2, 0], lattice[2, 1], lattice[2, 2]))

    def record_trajectory(self, frame, time=None, position=False, velocity=False, force=False, energy=None,
                          elapsed=None):

        if not self.trajectory.get(frame, False):
            self.trajectory[frame] = {}

        if time is not None:
            self.trajectory[frame]['t'] = time
        if position:
            positions = [at.position for at in self.atoms]
            self.trajectory[frame]['positions'] = np.array(positions)
        if velocity:
            velocities = [at.velocity for at in self.atoms]
            self.trajectory[frame]['velocities'] = velocities
        if force:
            forces = [at.force for at in self.atoms]
            self.trajectory[frame]['forces'] = np.array(forces)
        if energy is not None:
            self.trajectory[frame]['energy'] = energy
        if elapsed is not None:
            self.trajectory[frame]['elapsed'] = elapsed

    def get_species_mass(self, element):
        for atom in self.atoms:
            if atom.element == element:
                return atom.mass

        print("Warning! Element {} not found in structure".format(element))
        raise Exception("Tried to get mass of a species that didn't exist in structure.")

    @property
    def symbols(self):
        return [s[0] for s in self.get_positions()]

    @property
    def positions(self):
        return [at.position for at in self.atoms]

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def n_species(self):
        return len(self.species)

    def assert_boundary_conditions(self, verbosity=1):
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
        a1 = self.lattice[0]
        a2 = self.lattice[1]
        a3 = self.lattice[2]

        for atom in self.atoms:

            coords = np.dot(la.inv(self.lattice), atom.position[1])
            if any([coord > 1.0 for coord in coords]) or any([coord < 0.0 for coord in coords]):
                if verbosity == 4:
                    print('Atom positions before BC check:', atom.position)

                atom.position = a1 * (coords[0] % 1) + a2 * (coords[1] % 1) + a3 * (coords[2] % 1)
                if verbosity == 4:
                    print("BC updated position:", atom.position)


class Structure_Config(dict):
    """"
    Creates a Structure Config object, and runs validation on the
    possible parameters.
    Populated by dictionary loaded in from an input.yaml file.
    Used by the Structure class to instantiate itself.
    """

    def __init__(self, params, warn=True):

        self._params = ['lattice', 'alat', 'position', 'frac_pos', 'pos', 'fractional',
                        'unit_cell', 'pert_size', 'elements']
        self['lattice'] = None

        if params:
            self.update(params)

        self['elements'] = []
        self.positions = []

        check_list = {'alat': self.get('alat', False),
                      'position': self.get('frac_pos', False) or self.get('pos', False),
                      'lattice': self.get('lattice', False)}

        if warn and not all(check_list.values()):
            print('WARNING! Some critical parameters which are needed for structures'
                  ' to work are not present!!')
            for x in check_list.keys():
                if not check_list[x]:
                    print("Missing", x)
            raise Exception("Malformed input file-- structure parameters incorrect.")

        if self['lattice']:
            self['lattice'] = np.array(self['lattice'])
            self['unit_cell'] = self['alat'] * np.array([self['lattice'][0], self['lattice'][1], self['lattice'][2]])

        # TODO: ENSURE THAT FRACTIONAL POSITIONS ARE LOADED IN CORRECTLY
        # RIGHT NOW IT ISN'T RIGHT

        if self.get('pos', False) and self.get('frac_pos', False):
            print("Warning! Positions AND fractional positions were given--"
                  "This is not intended use! You must select one or the other in your input.")
            raise Exception("Fractional position AND Cartesian positions given.")

        if self['lattice'].shape != (3, 3):
            print("WARNING! Inappropriately shaped cell passed as input to structure!")
            raise Exception('Lattice has shape', np.shape(self['lattice']))

        if self.get('frac_pos', False):
            self.fractional = True
            self.positions = self['frac_pos']
        if self.get('pos', False):
            self.fractional = False
            self.positions = self['pos']

        else:
            self.fractional = True
            self.positions = self['frac_pos']

        for atom in self.positions:
            self['elements'].append(atom[0])

        # mass: force mass to type float if is integer or float
        if self.get('mass', False) and type(self.get('mass')) != type(list):
            self['mass'] = float(self['mass'])
        else:
            self['mass'] = 0

        # TODO: add support for list of mass

        if self.get('pert_size', False):
            for atom in self.positions:
                for pos in atom[1]:
                    pos += numpy.random.normal(0, scale=self['pert_size'])

        super(Structure_Config, self).__init__(self)

    def to_structure(self):

        mass_dict = {'H': 1.0, 'C': 12.01, "Al": 26.981539, "Si": 28.0855, 'O': 15.9994}

        atoms = []
        for n, pos in enumerate(self.positions):

            # Loop through position items defined in the yaml file

            if self.get('velocities', False):
                velocity = self['velocities'][n]
            else:
                velocity = (0, 0, 0)

            if self.get('forces', False):
                force = self['forces'][n]
            else:
                force = (0, 0, 0)

            if type(self['mass']) != type(list) and self['mass'] != 0:
                mass = self['mass']
            else:
                mass = mass_dict.get(pos[0], 1.0)

            atoms.append(Atom(position=list(pos[1]), velocity=list(velocity), force=list(force),
                              mass=float(mass), element=pos[0]))

        return Structure(atoms=atoms, alat=self['alat'], lattice=self['lattice'], fractional=self.fractional)


def ase_to_structure(struc, alat, fractional, perturb=0):
    """
    Quick helper function which turns an ASE structure
    to a PyFly one. Warning: You must specify if the structure is specified
    in fractional coordinates, and if a scaling factor by alat is necessary.
    You must also import ASE yourself.
    :param alat:
    :param fractional:
    :param struc: ASE Structure object
    :param fractional: Flag to handle if the atomic positions
            are in fractional coordinates
    :param perturb: Perturb atomic positions by a Gaussian with std.dev perturb
    :return:
    """
    positions = struc.get_positions()
    symbols = struc.get_chemical_symbols()
    lattice = struc.get_cell

    atoms = []
    for n in range(len(positions)):
        atoms.append(Atom(position=positions[n], element=symbols[n]))

    # TODO ADD THIS
    if perturb > 0:
        for at in atoms:
            for coord in range(3):
                at.position[coord] += random.normal(0, scale=perturb)
        pass

    return Structure(atoms, alat=alat, lattice=np.array(lattice), fractional=fractional)


# ------------------------------------------------------
#              ESPRESSO Config and Objects
# ------------------------------------------------------

# Object to load in and validate parsed QE settings
class QE_Config(dict):
    """
    Contains parameters which configure Quantum ESPRESSO pwscf runs,
    as well as the methods to implement them.
    """

    @property
    def pwin(self):
        return self['pwscf_input']

    def __init__(self, params=None, warn=False):

        if params is None:
            params = {}

        self._params = ['nk', 'system_name',
                        'pseudo_dir', 'outdir', 'pw_command', 'in_file',
                        'out_file', 'update_name', 'correction_folder',
                        'molecule', 'serial']

        super(QE_Config, self).__init__()

        if params:
            self.update(params)

        qe = self

        mandatory_params = ['nk', 'pw_command',
                            'in_file']
        mandatory_pw = {'CONTROL': ['pseudo_dir'],
                        'SYSTEM': ['ecutwfc', 'ecutrho']}

        # -------------------------
        # Check for missing mandatory parameters
        # -----------------------

        missing_mand = False
        if warn and not all([qe.get(param, False) for param in mandatory_params]):
            print('WARNING! Some critical parameters which are needed for QE to work are not present!')
            for param in mandatory_params:
                if not qe.get(param, False):
                    missing_mand = True
                    print("Missing parameter ", param, '\n')

        for card in ["CONTROL", "SYSTEM"]:
            if warn and not all([self['pwscf_input'][card].get(param, False) for param in mandatory_pw[card]]):
                print('WARNING! Some critical parameters which are needed for QE to work are not present!')
                for param in mandatory_pw[card]:
                    if not self.pwin[card].get(param, False):
                        missing_mand = True
                        print("Missing parameter in", card + ':' + param, '\n')

        if missing_mand:
            raise Exception("Missing necessary QE parameters.")

        # -------------------------
        # Check for missing default parameters
        # -----------------------

        missing_params = []
        default_qe = {'system_name': 'QE', 'pw_command': os.environ.get('PWSCF_COMMAND'),
                      'parallelization': {'np': 1, 'nk': 0, 'nt': 0, 'nd': 0, 'ni': 0}}

        # Do higher-level missing parameters

        for key in default_qe.keys():
            if key not in self.keys():
                missing_params.append(key)

        for key in missing_params:
            print("Achtung! Missing MD parameter, running model with default for: {}".format(key))
            self[key] = default_qe[key]

        # Do pwscf.input-level missing parameters
        default_pw = {'CONTROL': {'calculation': 'scf', 'disk_io': 'low', 'tprnfor': True,
                                  'wf_collect': False},
                      "ELECTRONS": {'diagonalization': 'david', 'mixing_beta': .5,
                                    'conv_thr': 1.0e-7}}
        missing_pw_params = {'CONTROL': [], "ELECTRONS": []}

        for card in default_pw.keys():
            for key in default_pw[card]:
                if key not in self.pwin[card].keys():
                    missing_pw_params[card].append(key)

            for key in missing_pw_params[card]:
                print("Achtung! Missing MD parameter, running model with default for: {}".format(key))
                self.pwin[card][key] = default_pw[card][key]

        # ----------------
        # Set up K points
        # ----------------

        nk = self['nk']
        if isinstance(nk, int) and (nk == 1 or nk == 0):
            option = 'gamma'

        elif isinstance(nk, list) and list([int(n) for n in nk]) == [1, 1, 1]:
            option = 'gamma'

        else:
            option = 'automatic'

        self.kpts = {'option': option, 'gridsize': [int(nk)] * 3 if (isinstance(nk, int) or isinstance(nk, float))

        else [nk[0], nk[1], nk[2]]}

        if type(nk) == type(list) and len(nk) > 3:
            self.kpts['offset'] = [nk[-3], nk[-2], nk[-1]]

        for s in ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL']:
            if not self['pwscf_input'].get(s, False):
                self['pwscf_input'][s] = {}

    def validate_against_structure(self, structure):
        """
        Tests to make sure that ESPRESSO will be able to run
        correctly for the structure which is provided.
        :param structure: Structure object
        :return bool: If ESPRESSO should be able to run with this structure
        """

        for species in structure.species:
            if species not in self['species_pseudo'].keys():
                print("WARNING! A pseudopotential file is not"
                      "specified for the species ", species)

        return True

    def as_dict(self):
        d = dict(self)
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        return d

    def run_espresso(self, structure, cnt=np.random.randint(1e10), augment_db=False):
        """
        Sets up the directory where pwscf is to be run; then, calls pw.x.
        Changes depending on if an augmentation run is being called i.e. one which is
        going to be a part of a ML Regression model re-training later.

        :param structure:       structure object.
        :param augment_db:      boolean, True if run will be a part of updated ML database
        :return:
        """

        if augment_db:

            # update db
            dirname = os.path.join(self['correction_folder'], 'db_update_' + str(cnt))
            prepare_dir(dirname)

        else:
            dirname = 'temprun'

        runpath = os.path.join(self.get('outdir', '.'), dirname)
        prepare_dir(runpath)
        self.write_pwscf_input(structure, runpath)

        # STILL PATCHY BELOW THIS LINE
        output_file = self.execute_qe_pwscf(runpath, runpath)

        output = self.parse_qe_pwscf_output(output_file)
        
        return output

    @staticmethod
    def parse_qe_pwscf_output(outfile):
        forces = []
        positions = []
        total_force = None
        pressure = None
        total_energy = np.nan
        kpoints = None
        volume = None

        # Flags to start collecting final coordinates
        final_coords = False
        get_coords = False

        with open(outfile, 'r') as outf:
            for line in outf:
                if line.lower().startswith('     pwscf'):
                    walltime = line.split()[-3] + line.split()[-2]

                if line.lower().startswith('     total force'):
                    total_force = float(line.split()[3]) * (13.605698066 / 0.529177249)

                if line.lower().startswith('!    total energy'):
                    total_energy = float(line.split()[-2]) * 13.605698066

                if line.lower().startswith('          total   stress'):
                    pressure = float(line.split('=')[-1])

                if line.lower().startswith('     number of k points='):
                    kpoints = float(line.split('=')[-1])

                if line.lower().startswith('     unit-cell volume'):
                    line = line.split('=')[-1]
                    line = line.split('(')[0]
                    line = line.strip()
                    volume = float(line)

                ## Chunk of code meant to get the final coordinates of the atomic positions
                if line.lower().startswith('begin final coordinates'):
                    final_coords = True
                    continue
                if line.lower().startswith('atomic_positions') and final_coords == True:
                    continue
                if line.lower().startswith('end final coordinates'):
                    final_coords = False
                if final_coords == True and line.split() != []:
                    positions.append([line.split()[0], [float(line.split()[1]),
                                                        float(line.split()[2]), float(line.split()[3])]])

                if line.find('force') != -1 and line.find('atom') != -1:
                    line = line.split('force =')[-1]
                    line = line.strip()
                    line = line.split(' ')
                    # print("Parsed line",line,"\n")
                    line = [x for x in line if x != '']
                    temp_forces = []
                    for x in line:
                        temp_forces.append(float(x))
                    forces.append(list(temp_forces))

        if total_energy == np.nan:
            print("WARNING! ")
            raise Exception("Quantum ESPRESSO parser failed to read the file {}. Run failed.".format(outfile))

        result = {'energy': total_energy, 'kpoints': kpoints, 'volume': volume, 'positions': positions}
        if forces:
            result['forces'] = forces
        if total_force is not None:
            result['total_force'] = total_force
        if pressure is not None:
            result['pressure'] = pressure
        return result

    def execute_qe_pwscf(self, inpath, outdir, target='pwscf.out'):

        outpath = os.path.join(outdir, target)

        pw_command = self.get('pw_command', os.environ['PWSCF_COMMAND'])

        if self.get('serial', False):
            pw_command = "{} < {} > {}".format(
                pw_command, os.path.join(inpath, self['in_file']), outpath)

        else:
            par_string = ''
            for par, val in self['parallelization'].items():
                par_string += '-{} {}'.format(par, val) if val != 0 else ''
            pw_command = 'mpirun {0} -npool {1} < {2} > {3}'.format(pw_command, par_string, inpath, out_file)

        run_command(pw_command)
        return outpath

    def run_scf_from_text(self, scf_text, npool, out_file='pw.out', in_file='pw.in'):

        # call qe
        qe_command = 'mpirun {0} -npool {1} < {2} > {3}'.format(self['pw_loc'], npool, in_file, out_file)
        run_command(qe_command)

    @staticmethod
    def qe_value_map(value):
        """
    	Function used to interpret correctly values for different
    	fields in a Quantum Espresso input file (i.e., if the user
    	specifies the string '1.0d-4', the quotes must be removed
    	when we write it to the actual input file)
    	:param: a string
    	:return: formatted string to be used in QE input file
    	"""
        if isinstance(value, bool):
            if value:
                return '.true.'
            else:
                return '.false.'
        elif isinstance(value, (float, np.float)) or isinstance(value, (int, np.int)):
            return str(value)
        elif isinstance(value, str):
            return "'{}'".format(value)
        else:
            print("Strange value ", value)
            raise ValueError

    def write_pwscf_input(self, structure, runpath):
        """Make input param string for PW
        args:
        structure (Structure object)
        runpath (str): path to where to write pwscf input file
        """
        # automatically fill in missing values

        self.pwin['SYSTEM']['ntyp'] = structure.n_species
        self.pwin['SYSTEM']['nat'] = structure.n_atoms
        self.pwin['SYSTEM']['ibrav'] = 0

        # Write the main input block
        inptxt = ''
        for namelist in ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL']:
            inptxt += '&{}\n'.format(namelist)
            for key, value in self.pwin[namelist].items():
                inptxt += '	{} = {}\n'.format(key, self.qe_value_map(value))
            inptxt += '/ \n'

        # --------------------------
        # write the K_POINTS block
        # --------------------------
        if self.kpts['option'] == 'automatic':
            inptxt += 'K_POINTS {automatic}\n'

        if self.kpts['option'] == 'gamma':
            inptxt += "K_POINTS {gamma}\n"
        else:
            inptxt += ' {:d} {:d} {:d}'.format(*self.kpts['gridsize'])

            if self.kpts.get('offset', False):
                inptxt += '  1 1 1\n'
            else:
                inptxt += '  0 0 0\n'

        # write the ATOMIC_SPECIES block
        inptxt += 'ATOMIC_SPECIES\n'
        for elem in structure.species:
            inptxt += '  {} {} {}\n'.format(elem, structure.get_species_mass(elem), self['species_pseudo'][elem])

        # Write the CELL_PARAMETERS block
        inptxt += 'CELL_PARAMETERS {angstrom}\n'
        for vector in structure.lattice:
            inptxt += ' {} {} {}\n'.format(vector[0], vector[1], vector[2])

        # Write the ATOMIC_POSITIONS in crystal coords
        inptxt += 'ATOMIC_POSITIONS {angstrom}\n'
        for atom in structure:
            inptxt += '  {} {:1.5f} {:1.5f} {:1.5f} \n'.format(atom.element, *atom.position)

        infile = os.path.join(runpath, self['in_file'])

        f = open(infile, 'w')
        f.write(inptxt)
        f.close()

        return infile


# ------------------------------------------------------
#              High-performance Computing Config
# ------------------------------------------------------

class HPC_Config(dict):
    def __init__(self, params, warn=True):
        self.warn = warn
        self._params = ['cores', 'nodes', 'partition', 'time']

        if params:
            self.update(params)

        super(HPC_Config, self).__init__(params)


# ------------------------------------------------------------
#       General Configuration (Loading and setting up)
# ------------------------------------------------------------


def load_config_yaml(path, verbose=True):
    """"
    Loads configuration from input.yaml,
    """
    if not os.path.isfile(path) and verbose:
        raise OSError('Configuration file does not exist.')

    with open(path, 'r') as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

        return out


# TODO Make this a nice all-in-one loader
def setup_configs(path, verbose=True):
    setup_dict = load_config_yaml(path, verbose=verbose)

    if 'md_params' in setup_dict.keys():
        md = MD_Config(setup_dict['md_params'])
    else:
        md = MD_Config({})

    if 'qe_params' in setup_dict.keys():
        qe = QE_Config(setup_dict['qe_params'])
    else:
        qe = QE_Config({})

    if 'structure_params' in setup_dict.keys():
        structure = Structure_Config(setup_dict['structure_params']).to_structure
    else:
        print("Warning-- Input file does not contain any structure parameters. Will not"
              "return structure parameters.")
        structure = Structure_Config({})

    if 'ml_params' in setup_dict.keys():
        ml = ml_config(setup_dict['ml_params'])

        if ml_config['regression_model'] == 'GP':
            pass
            # ml= GaussianProcess() #TODO: integrate this with Simon's new classes

    else:
        ml = ml_config(params=setup_dict['ml_params'], print_warn=True)

    return structure, md,


def main():
    # load from config file
    config = load_config_yaml('input.yaml')
    print(type(config))

    # # # set configs
    # qe_conf = QE_Config(config['qe_params'], warn=True)
    # print(qe_conf)
    #
    # structure = Structure_Config(config['structure_params']).to_structure()
    # print(structure)
    #
    # print(qe_conf.run_espresso(structure))
    #
    # # ml_fig = ml_config(params=config['ml_params'], print_warn=True)
    # # print(ml_fig)
    #
    # md_fig = MD_Config(params=config['md_params'], warn=True)
    # print(md_fig)

    # hpc_fig = HPC_Config(params=config['hpc_params'])


if __name__ == '__main__':
    main()
