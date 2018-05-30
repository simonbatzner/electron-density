from labutil.src.objects import *
import numpy


class PWscf_inparam(Param):
    """
    Data class containing parameters for a Quantum Espresso PWSCF calculation
    it does not include info on the cell itself, since that will be taken from a Struc object
    """
    pass


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
    elif isinstance(value, (float, numpy.float)) or isinstance(value, (int, numpy.int)):
        return str(value)
    elif isinstance(value, str):
        return "'{}'".format(value)
    else:
        print("Strange value ", value)
        raise ValueError


def write_pwscf_input(runpath, params, struc, kpoints, pseudopots, constraint=None):
    """Make input param string for PW"""
    # automatically fill in missing values
    pcont = params.content
    pcont['SYSTEM']['ntyp'] = struc.n_species
    pcont['SYSTEM']['nat'] = struc.n_atoms
    pcont['SYSTEM']['ibrav'] = 0
    # Write the main input block
    inptxt = ''
    for namelist in ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL']:
        inptxt += '&{}\n'.format(namelist)
        for key, value in pcont[namelist].items():
            inptxt += '    {} = {}\n'.format(key, qe_value_map(value))
        inptxt += '/ \n'
    # write the K_POINTS block
    if kpoints.content['option'] == 'automatic':
        inptxt += 'K_POINTS {automatic}\n'
    inptxt += ' {:d} {:d} {:d}'.format(*kpoints.content['gridsize'])
    if kpoints.content['offset']:
        inptxt += '  1 1 1\n'
    else:
        inptxt += '  0 0 0\n'

    # write the ATOMIC_SPECIES block
    inptxt += 'ATOMIC_SPECIES\n'
    for elem, spec in struc.species.items():
        inptxt += '  {} {} {}\n'.format(elem, spec['mass'], pseudopots[elem].content['name'])

    # Write the CELL_PARAMETERS block
    inptxt += 'CELL_PARAMETERS {angstrom}\n'
    for vector in struc.content['cell']:
        inptxt += ' {} {} {}\n'.format(*vector)

    # Write the ATOMIC_POSITIONS in crystal coords
    inptxt += 'ATOMIC_POSITIONS {angstrom}\n'
    for index, positions in enumerate(struc.content['positions']):
        inptxt += '  {} {:1.5f} {:1.5f} {:1.5f}'.format(positions[0], *positions[1])
        if constraint and constraint.content['atoms'] and str(index) in constraint.content['atoms']:
            inptxt += ' {} {} {} \n'.format(*constraint.content['atoms'][str(index)])
        else:
            inptxt += '\n'

    infile = TextFile(path=os.path.join(runpath.path, 'pwscf.in'), text=inptxt)
    infile.write()
    return infile


def run_qe_pwscf(struc, runpath, pseudopots, params, kpoints, constraint=None, ncpu=1):
    pwscf_code = ExternalCode({'path': os.environ['PWSCF_COMMAND']})
    prepare_dir(runpath.path)
    infile = write_pwscf_input(params=params, struc=struc, kpoints=kpoints, runpath=runpath,
                               pseudopots=pseudopots, constraint=constraint)
    outfile = File({'path': os.path.join(runpath.path, 'pwscf.out')})
    pwscf_command = "mpirun -np {} {} < {} > {}".format(ncpu, pwscf_code.path, infile.path, outfile.path)
    run_command(pwscf_command)
    return outfile


def parse_qe_pwscf_output(outfile):

    forces=[]

    with open(outfile.path, 'r') as outf:
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
                kpoints  = float(line.split('=')[-1])

            if line.lower().startswith('     unit-cell volume'):
                line=line.split('=')[-1]
                line=line.split('(')[0]
                line=line.strip()
                volume   = float(line)

            if 'force' in line and 'atom' in line:
                line=line.split('force =')[-1]
                line=line.strip()
                line=line.split(' ')
                #print("Parsed line",line,"\n") 
                line=[x for x in line if x!='']
                temp_forces=[]
                for x in line:
                    temp_forces.append(float(x))
                forces.append(list(temp_forces))

            
            if 'subroutine stress' in line:
                grab_forces=False


        

    result = {'energy': total_energy, 'force': total_force, 'pressure': pressure, 'kpoints':kpoints, 'forces': forces, 'volume': volume}
    return result


def parse_qe_relax_output(outfile):

    forces=[]
    grab_params=True
    param_counter=0
    cell_params=[[0,0,0],[0,0,0],[0,0,0]]

    with open(outfile.path, 'r') as outf:
        for line in outf:
            if line.lower().startswith('     pwscf'):
                walltime = line.split()[-3] + line.split()[-2]
            if line.lower().startswith('     total force'):
                total_force = float(line.split()[3]) * (13.605698066 / 0.529177249)
            if line.lower().startswith('!    total energy'):

                total_energy = float(line.split()[-2]) * 13.605698066
            if line.lower().startswith('          total   stress'):
                pressure = float(line.split()[-1])
            if line.lower().startswith('     number of k points='):
                kpoints  = float(line.split('=')[-1])

            if 'force' in line and 'atom' in line:
                line=line.split('force =')[-1]
                line=line.strip()
                line=line.split(' ')
                #print("Parsed line",line,"\n") 
                line=[x for x in line if x!='']
                temp_forces=[]
                for x in line:
                    temp_forces.append(float(x))
                forces.append(list(temp_forces))

            
            if 'subroutine stress' in line:
                grab_forces=False

            if 'CELL PARAMETERS' in line:
                grab_params=True
                param_counter=0
        
            if 'ATOMIC_POSITIONS' in line:
                grab_params=False

            if grab_params==True and param_counter<3:
                line=line.strip()
                line=line.split(' ')
                cell_params[param_counter] = [x for x in line if x!='']
                param_counter+=1
                continue



    result = {'energy': total_energy, 'force': total_force, 'pressure': pressure, 'kpoints':kpoints, 'forces': forces,'cell_params':cell_params}
    return result
