import sys
import os

import yaml
import numpy as np
import numpy.random as random
import pprint
import time as time
import numpy.linalg as la
from Jon_Production.utility import write_file, run_command


#TODO: Move this to utility file once we're done with everything else
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

#TODO: Move this to utility file once we're done with everything else here
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__





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

    def __init__(self, params,warn=True):

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
                      'dt':.1,
                      'mass': 0,
                      'fd_dx': .01,
                      'verbosity': 1,
                      'assert_boundaries': False,
                      'timestep_method': 'Verlet',
                      'fd_accuracy': 4,
                      'energy_or_force': 'Force'}

        # If value was not specified, load in default values.

        if self.get('frames',False) and self.get('tf',False) and self.get('dt', False) and warn:
            print("WARNING! Frames and final time specified;"
                  " these two parameters cannot both exist. Pick either a number of timesteps"
                  " or an automatically computed computer of timesteps.")
            raise Exception("Number of frames and final time both specified.")

        if self.get('frames',False) and self.get('tf',False) and self.get('dt', False):
            print("WARNING! Frames and final time specified;"
                  " these two parameters cannot both exist. Pick either a number of timesteps"
                  " or an automatically computed computer of timesteps.")
            raise Exception("Number of frames and final time both specified.")

        self.time  = 0 + self.get('ti',0)
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
        self.trajectory = {} #keys: time rounded to 3 decimal places, values: {'position':[array], 'force':[array]}

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

        super(Structure, self).__init__(self.atoms)

    def get_pprint_atoms(self):

        fractional = self.fractional
        report=''
        if fractional:
            for n, at in enumerate(self.atoms):
                report+="{}:{} ({},{},{}) \n".format(n, at.element, at.position[0], at.position[1], at.position[2])
        else:
            for n, at in enumerate(self.atoms):
                report+="{}:{} ({},{},{}) \n".format(n, at.element, at.position[0], at.position[1], at.position[2])

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

    def record_trajectory(self,time,position=False,force=False):
        t = np.round(time,3)
        for at in self:
            if not at.trajectory.get(t, False):
                at.trajectory[t] = {}
            if position:
                at.trajectory[t]['position'] = np.array(at.position)
            if force:
                at.trajectory[t]['force'] = np.array(at.force)



    def assert_boundary_conditions(self,verbosity=1):
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

            coords = np.dot(la.inv(self.lattice), atom.position)
            if any([coord > 1.0 for coord in coords]) or any([coord < 0.0 for coord in coords]):
                if verbosity == 4: print('Atom positions before BC check:', atom.position)

                atom.position = a1 * (coords[0] % 1) + a2 * (coords[1] % 1) + a3 * (coords[2] % 1)
                if verbosity == 4: print("BC updated position:", atom.position)



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

        if self.get('pos', False) and self.get('frac_pos', False):
            print("Warning! Positions AND fractional positions were given--"
                  "This is not intended use! You must select one or the other in your input.")
            raise Exception("Fractional position AND Cartesian positions given.")

        if self['lattice'].shape != (3, 3):
            print("WARNING! Inappropriately shaped cell passed as input to structure!")
            raise Exception('Lattice has shape', np.shape(self['lattice']))

        if self.get('frac_pos', False):
            self.fractional = True
            self.positions=self['frac_pos']
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

            if self.get('velocities',False):
                velocity = self['velocities'][n]
            else:
                velocity = (0, 0, 0)

            if self.get('forces',False):
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




# ------------------------------------------------------
#              ESPRESSO Config and Objects
# ------------------------------------------------------

# Object to load in and validate parsed QE settings



# Object to handle QE
class QE_Config(dict):
    """
    Contains parameters which configure Quantum ESPRESSO pwscf runs,
    as well as the methods to implement them.
    """

    def __init__(self, params={}, warn=False):

        super(QE_Config, self).__init__()

        if params:
            self.update(params)
        qe = self


        if not (qe.get('system_name', False)): self['system_name'] = 'QE'
        if not (qe.get('pw_command', False)): self['pw_command'] = os.environ.get('PWSCF_COMMAND')
        if not (qe.get('parallelization', False)): self['parallelization'] = {'np': 1, 'nk': 0, 'nt': 0, 'nd': 0,
                                                                              'ni': 0}

        if warn and not all([
            qe.get('ecut', False),
            qe.get('nk', False),
            qe.get('sc_dim', False),
            qe.get('pw_command', False),
            qe.get('pseudo_dir', False),
            qe.get('in_file', False)]):
            print('WARNING! Some critical parameters which are needed for QE to work are not present!')

    def as_dict(self):
        d = dict(self)
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        return d

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
                                   parallelization=self.parallelization)
        output = parse_qe_pwscf_output(outfile=output_file)

        with open(runpath.path + '/en', 'w') as f:
            f.write(str(output['energy']))
        with open(runpath.path + '/pos', 'w')as f:
            for pos in [atom.position for atom in atoms]:
                f.write(str(pos) + '\n')

        return output

    def create_scf_input(self):
        """
        Jon V's version of the PWSCF formatter.
        Works entirely based on internal settings.
        """

        scf_text = """ &control
            calculation = 'scf'
            pseudo_dir = '{0}'
            outdir = '{1}'
            tprnfor = .true.
         /
         &system
            ibrav= 0
            nat= {2}
            ntyp= 1
            ecutwfc ={3}
            nosym = .true.
         /
         &electrons
            conv_thr =  1.0d-10
            mixing_beta = 0.7
         /
        ATOMIC_SPECIES
         Si  28.086  Si.pz-vbc.UPF
        {4}
        {5}
        K_POINTS automatic
         {6} {6} {6}  0 0 0
            """.format(self['pseudo_dir'], self['outdir'],
                       self['nat'], self['ecut'], self['cell'], self['pos'], self['nk'])
        return scf_text

    def run_scf_from_text(self, scf_text, npool, out_file='pw.out', in_file='pw.in'):

        # write input file
        write_file(in_file, scf_text)

        # call qe
        qe_command = 'mpirun {0} -npool {1} < {2} > {3}'.format(self['pw_loc'], npool, in_file, out_file)
        run_command(qe_command)




# ------------------------------------------------------
#              High-performance Computing Config
# ------------------------------------------------------

class HPC_Config(dict):

    def __init__(self, params, warn=True):

        self._params = ['cores', 'nodes', 'partition', 'time']

        if params:
            self.update(params)

        super(HPC_Config,self).__init__(params)


# ------------------------------------------------------------
#       General Configuration (Loading and setting up)
# ------------------------------------------------------------


def load_config_yaml(path, verbose=True):
    if not os.path.isfile(path) and verbose:
        raise OSError('Configuration file does not exist.')

    with open(path, 'r') as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return out


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
    print(config)

    # # set configs
    qe_conf = QE_Config(config['qe_params'], warn=True)
    print(qe_conf)

    structure = Structure_Config(config['structure_params'])
    print(struc_fig)
    print(struc_fig._params)

    ml_fig = ml_config(params=config['ml_params'], print_warn=True)
    print(ml_fig)

    md_fig = MD_Config(params=config['md_params'],warn=True)
    print(md_fig)

    hpc_fig = HPC_Config(params=config['hpc_params'])


if __name__ == '__main__':
    main()
