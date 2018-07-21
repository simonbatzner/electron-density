#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name, too-many-arguments

"""" Input parser

Steven Torrisi, Simon Batzner
"""

import os

import yaml
import numpy as np
import numpy.random
import pprint

from ase import Atoms

from Jon_Production.utility import write_file, run_command


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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class md_config(dict):
    """
    Creates an md_params object.

    Args:
        params (dict): A set of input parameters as a dictionary.
    """

    def __init__(self, params):
        super(md_config, self).__init__()

        if params:
            self.update(params)


class ml_config(dict):
    """
    Creates an ml_params object.
    """

    def __init__(self, params, print_warn=True):
        """
        Init
        :param params:      dict, input dictionary to init from
        :param print_warn:  boolean, whether to print warning if certain parameters are missing
        """

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


class structure_config(dict):
    """
    Holds info on atomic structure
    """

    def __init__(self, params, warn=True):
        """
        Init
        :param params:  dict, input dictionary to init from
        :param warn:    boolean, whether to print warning if certain parameters are missing
        """

        self._params = ['lattice', 'alat', 'position', 'frac_pos', 'pos', 'fractional',
                        'unit_cell', 'pert_size', 'elements']
        self['lattice'] = None

        if params:
            self.update(params)

        # super(structure_config, self).__init__()

        self['elements'] = []
        self.positions = []

        check_list = {'alat': self.get('alat', False),
                      'position': self.get('frac_pos', False) or self.get('pos', False),
                      'lattice': self.get('lattice', False)}

        if warn and not all(check_list.values()):

            print('WARNING! Some critical parameters which are needed for structures'
                  ' to work are not present!!')

            for x in check_list.keys():
                if not check_list[x]: print("Missing", x)

            raise Exception("Malformed input file-- structure parameters incorrect.")

        if self['lattice']:
            self['unit_cell'] = self['alat'] * np.array([self['lattice'][0], self['lattice'][1], self['lattice'][2]])

        if self.get('pos', False) and self.get('frac_pos', False):
            print("Warning! Positions AND fractional positions were given--"
                  "This is not intended use! You must select one or the other in your input.")

            raise Exception("Fractional position AND Cartesian positions given.")

        if self.get('pos', False):
            self.fractional = False
            self.positions = self['pos']

        else:
            self.fractional = True
            self.positions = self['frac_pos']

        for atom in self.positions:
            self['elements'].append(atom[0])

        if self.get('pert_size'):
            for atom in self.positions:
                for pos in atom[1]:
                    pos += numpy.random.normal(0, scale=self['pert_size'])

        super(structure_config, self).__init__(self)


def load_config(path, verbose=True):
    """
    Load parameters from input.yaml
    :param path:        str, path of configuration file
    :param verbose:     boolean, verbosity
    :return:            dict, configuration parameters
    """

    if not os.path.isfile(path) and verbose:
        raise OSError('Configuration file does not exist.')

    with open(path, 'r') as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return out


def setup_configs(path, verbose=True):
    """
    Set subconfigurations from input.yaml
    :param path:        str, path of configuration file
    :param verbose:     boolean, verbosity
    :return:
    """

    setup_dict = load_config(path, verbose=verbose)

    # if 'md_params' in setup_dict.keys():
    #     md = md_params.from_dict(setup_dict['md_params'])
    # else:
    #     md = md_params.from_dict({})

    if 'qe_params' in setup_dict.keys():
        qe = qe_config.from_dict(setup_dict['qe_params'])
    else:
        qe = qe_config.from_dict({})

    if 'structure_params' in setup_dict.keys():
        structure = structure_config(setup_dict['structure_params'])
    else:
        structure = structure_config({})

    if 'ml_params' in setup_dict.keys():
        ml = ml_config(setup_dict['ml_params'])

        if ml_config['regression_model'] == 'GP':
            pass
            # ml= GaussianProcess() #TODO: integrate this with Simon's new classes

    else:
        ml = ml_config(params=setup_dict['ml_params'], print_warn=True)


class qe_config(dict):
    """
    Contains parameters which configure Quantum ESPRESSO pwscf runs,
    as well as the methods to implement them.
    """

    def __init__(self, params={}, warn=False):
        """
        Init
        :param params:  dict, input dictionary to init from
        :param warn:    boolean, whether or not to print warning if params are missing
        """
        super(qe_config, self).__init__()

        # init
        if params:
            self.update(params)

        qe = self
        self.correction_folder = self.get('correction_folder')

        # check for missing params
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

    # @classmethod
    # def from_dict(cls, d):
    #    return qe_config({k: v for k, v in d.items() if k not in ("@module",
    #                                                          "@class")})

    def get_correction_number(self):

        walk_result = list(os.walk(self.correction_folder))
        folders_in_correction_folder = walk_result[0][1]

        steps = [fold for fold in folders_in_correction_folder if self.system_name + "_step_" in fold]

        if len(steps) >= 1:
            stepvals = [int(fold.split('_')[-1]) for fold in steps]
            correction_number = max(stepvals)
            print(correction_number)

        else:
            return 0

        return correction_number + 1

    def run_espresso(self, atoms, cell, iscorrection=False):
        """
        Run Quantum Espresso
        :param atoms:
        :param cell:
        :param iscorrection:
        :return:
        """
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

            # print("rolling with correction number", qe_config.correction_number)
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

    def create_scf_input(self):
        """
        Create QE SCF input file

        :return     str, input text for QE
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
            """.format(self['pseudo_dir'], self['outdir'], \
                       self['nat'], self['ecut'], self['cell'], self['pos'], self['nk'])
        return scf_text

    def run_scf_from_text(self, scf_text, npool, out_file='pw.out', in_file='pw.in'):
        """
        Write QE input and run QE SCF from scf text file
        """
        # write input file
        write_file(in_file, scf_text)

        # call qe
        qe_command = 'mpirun {0} -npool {1} < {2} > {3}'.format(self['pw_loc'], npool, in_file, out_file)
        run_command(qe_command)


def main():

    # load from config file
    config = load_config('input.yaml')
    print(type(config))

    # set configs
    pprint.pprint(config['qe_params'])
    qe_fig = qe_config(config['qe_params'], warn=True)
    pprint.pprint(qe_fig)
    print(qe_fig.get_correction_number())


if __name__ == '__main__':
    main()
