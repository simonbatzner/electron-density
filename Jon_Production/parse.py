import yaml
import sys
import os
from mson import MSONable
import os.path
from utility import write_file, run_command
## The form of the below code borrows from Pymatgen,  http://pymatgen.org/index.html
## and their io classes. The MSONable class is borrowed from Monty: http://guide.materialsvirtuallab.org/monty/_modules/monty/json.html

class md_params(MSONable):
    """
    Creates an md_params object.

    Args:
        params (dict): A set of input parameters as a dictionary.
    """
    def __init__(self,params):
        super(md_params, self).__init__()


class qe_config(MSONable):
    """
    Creates an md_params object.

    Args:
        params (dict): A set of input parameters as a dictionary.
    """
    def __init__(self,params):
        super(qe_config, self).__init__()


class ml_config(MSONable):
    """
    Creates an md_params object.

    Args:
        params (dict): A set of input parameters as a dictionary.
    """
    def __init__(self,params):
        super(ml_config, self).__init__()


class structure_config(MSONable):

    def __init__(self,params):
        self['lattice'] = None
        super(structure_config, self).__init__()

        if self['lattice']:
            self['unit_cell']= np.array(self['lattice']['vec1'],self['lattice']['vec2'],self['lattice']['vec3'])


def load_config(path,verbose=True):

    if not os.path.isfile(path) and verbose:
        raise OSError('Configuration file does not exist.')

    with open(path, 'r') as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    return out

def setup_configs(path,verbose=True):

    setup_dict = load_config(path, verbose=verbose)


    if 'md_params' in setup_dict.keys():
        md= md_params.from_dict(setup_dict['md_params'])
    else:
        md = md_params.from_dict({})

    if  'qe_params' in setup_dict.keys():
        qe = qe_config.from_dict(setup_dict['qe_params'])
    else:
        qe = qe_config.from_dict({})

    if 'structure_params' in setup_dict.keys():
        structure = structure_config(setup_dict['structure_params'])
    else:
        structure = structure_config({})

    if 'ml_params' in setup_dict.keys():
        ml = ml_config(setup_dict['ml_params'])
        if ml_config['regression_model']=='GP':
            return GaussianProcess()
    else:
        ml = ml_config({})





class qe_config(MSONable):

    """
    Contains parameters which configure Quantum ESPRESSO pwscf runs,
    as well as the methods to implement them.
    """
    def __init__(self, params={},warn=False):

        super(qe_config, self).__init__()

        qe = self
        if not(qe.get('system_name',False)): self['system_name'] = 'QE'
        if not(qe.get('pw_command',False)): self['pw_command'] = os.environ.get('PWSCF_COMMAND')
        if not(qe.get('parallelization',False)): self['parallelization'] = {'np':1,'nk':0,'nt':0,'nd':0,'ni':0}

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

    @classmethod
    def from_dict(cls, d):
        return qe_config({k: v for k, v in d.items() if k not in ("@module",
                                                              "@class")})

    def get_correction_number(self):
        folders_in_correction_folder = list(os.walk(self.correction_folder))[0][1]

        steps = [fold for fold in folders_in_correction_folder if self.system_name + "_step_" in fold]

        if len(steps) >= 1:
            stepvals = [int(fold.split('_')[-1]) for fold in steps]
            correction_number = max(stepvals)
        else:
            return 0
        return correction_number + 1

    def run_espresso(self, atoms, cell, iscorrection = False):

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
            """.format(self['pseudo_dir'], self['outdir'], \
                       self['nat'], self['ecut'], self['cell'], self['pos'], self['nk'])
        return scf_text

    def run_scf_from_text(self,scf_text, npool, out_file='pw.out', in_file='pw.in'):

        # write input file
        write_file(in_file, scf_text)

        # call qe
        qe_command = 'mpirun {0} -npool {1} < {2} > {3}'.format(self['pw_loc'], npool, in_file, out_file)
        run_command(qe_command)


a = load_config('input.yaml')
print(a)
b=qe_config(params=a['qe_params'])
