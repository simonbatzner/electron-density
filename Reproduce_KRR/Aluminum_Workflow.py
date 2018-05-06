from labutil.src.plugins.lammps import *
from labutil.src.plugins.pwscf import *
from ase.spacegroup import crystal
from ase.build import *
from ase.io import write
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import scipy as sp
from scipy.fftpack import *

def make_struc(size):
    """
    Creates the crystal structure using ASE.
    :param size: supercell multiplier
    :return: structure object converted from ase
    """
    alat = 4.10
    unitcell = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90])
    multiplier = numpy.identity(3) * size
    supercell = make_supercell(unitcell, multiplier)
    structure = Struc(ase2struc(supercell))
    return structure

def compute_dynamics_NVT(size, timestep, nsteps, temperature, tout):
    """
    Make an input template and select potential and structure, and input parameters.
    Return a pair of output file and RDF file written to the runpath directory.
    """
    intemplate = """
    # ---------- Initialize simulation ---------------------
    units metal
    atom_style atomic
    dimension  3
    boundary   p p p
    read_data $DATAINPUT

    pair_style eam/alloy
    pair_coeff * * $POTENTIAL  Al

    velocity  all create $TEMPERATURE 87287 dist gaussian

    # ---------- Describe computed properties------------------
    compute msdall all msd
    thermo_style custom step pe ke etotal temp press density c_msdall[4]
    thermo $TOUTPUT

    # ---------- Specify ensemble  ---------------------
    #fix  1 all nve
    fix  1 all nvt temp $TEMPERATURE $TEMPERATURE $TDAMP

    # --------- Compute RDF ---------------
    compute rdfall all rdf 100 1 1
    fix 2 all ave/time 1 $RDFFRAME $RDFFRAME c_rdfall[*] file $RDFFILE mode vector

    # --------- Dump and Run -------------
    dump myDump all atom $TOUTPUT $DUMPPATH
    timestep $TIMESTEP
    run $NSTEPS
    """

    potential = ClassicalPotential(ptype='eam', element='Al', name='Al_zhou.eam.alloy')
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Al_Data/LAMMPS", "size_" + str(size)))
    dumppath = os.environ['WORKDIR'] + '/Al_Data/LAMMPS/size_' + str(size)+'/dump.atom'
    struc = make_struc(size=size)
    inparam = {
        'TEMPERATURE': temperature,
        'NSTEPS': nsteps,
        'TIMESTEP': timestep,
        'TOUTPUT': tout,                 # how often to write thermo output
        'TDAMP': 50 * timestep,       # thermostat damping time scale
        'RDFFRAME': int(nsteps / 4),   # frames for radial distribution function
        'DUMPPATH': dumppath
    }
    outfile = lammps_run(struc=struc, runpath=runpath, potential=potential,
                                  intemplate=intemplate, inparam=inparam)
    output = parse_lammps_thermo(outfile=outfile)
    rdffile = get_rdf(runpath=runpath)
    rdfs = parse_lammps_rdf(rdffile=rdffile)
    return output, rdfs

# create supercell with customized positions
def struc_cust(size, alat, pos_array, pos_index):
    # create default supercell
    unitcell = crystal('Al', [(0, 0, 0)], spacegroup=225, \
                       cellpar=[alat, alat, alat, 90, 90, 90])
    multiplier = numpy.identity(3) * size
    supercell = make_supercell(unitcell, multiplier)
    structure = Struc(ase2struc(supercell))
    
    # modify the positions
    for n in range(len(structure.positions)):
        curr_pos = size*alat*pos_array[n,:,pos_index]
        structure.positions[n][1] = curr_pos.tolist()
        
    return structure

def compute_energy(alat, nk, ecut, struc, ncpu=2):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    potname = 'Al.pz-vbc.UPF'
    potpath = os.path.join(os.environ['ESPRESSO_PSEUDO'], potname)
    pseudopots = {'Al': PseudoPotential(path=potpath, ptype='ncpp', element='Al',
                                        functional='PZ', name=potname)}
    kpts = Kpoints(gridsize=[nk, nk, nk], option='automatic', offset=False)
    dirname = 'Al_Files'
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Al_Data", dirname))
    input_params = PWscf_inparam({
        'CONTROL': {
            'prefix': 'Al',
            'calculation': 'scf',
            'pseudo_dir': os.environ['ESPRESSO_PSEUDO'],
            'outdir': runpath.path,
            'wfcdir': runpath.path,
            'tstress': True,
            'tprnfor': True,
            'disk_io': 'medium',
            'wf_collect': True
        },
        'SYSTEM': {
            'ecutwfc': ecut,
            'ecutrho': ecut * 8,
            'occupations': 'smearing',
            'smearing': 'mp',
            'degauss': 0.02
             },
        'ELECTRONS': {
            'diagonalization': 'david',
            'mixing_beta': 0.5,
            'conv_thr': 1e-7,
        },
        'IONS': {},
        'CELL': {},
        })

    output_file = run_qe_pwscf(runpath=runpath, struc=struc,  pseudopots=pseudopots,
                               params=input_params, kpoints=kpts, ncpu=ncpu)
    output = parse_qe_pwscf_output(outfile=output_file)
    return output

def quick_density_gen():
    work_dir = Dir(path=os.path.join(os.environ['WORKDIR'], "Al_Data"))

    os.chdir(work_dir.path)

    dirname = 'Al_Files'

    input_text="""
     &inputpp
        prefix  = 'Al'
        outdir = '"""+work_dir.path+'/'+dirname + """'
        filplot = './"""+dirname+"""/Alrho'
        plot_num= 0
     /
     &plot
        nfile = 1
        filepp(1) = './""" + dirname + """/Alrho'
        weight(1) = 1.0
        iflag = 3
        output_format = 6
        fileout = './""" + dirname + """/Al.rho.dat'
        e1(1) =1.0, e1(2)=0.0, e1(3) = 0.0,
        e2(1) =0.0, e2(2)=1.0, e2(3) = 0.0,
        e3(1) =0.0, e3(2)=0.0, e3(3) = 1.0,
    /
    """

    with open('Al_rho_temp.pp','w') as f:
        f.write(input_text)
    os.system( os.environ["PP_COMMAND"] + " < Al_rho_temp.pp")
        
def rho_to_numpy(file_path,natoms=4):
    with open(file_path,'r') as f:
        thelines=f.readlines()

    #Parse the first 6 lines by default (the slice is exclusive ,hence :7)
    # Assuming we have 2 lines of comment header,
    # 3 lines of volumetric information,
    #
    header=thelines[0:6+natoms]
    body = thelines[6+natoms:]

    header=[line.strip() for line in header]
    ## ASSUMING CUBIC LATTICE FOR NOW/MOLECULE
    nx = int(header[3].split()[0])
    dx = float(header[3].split()[1])


    ny = int(header[4].split()[0])
    dy = float(header[4].split()[2])

    nz = int(header[5].split()[0])
    dz = float(header[5].split()[3])


    array=np.empty(shape=(nx,ny,nz))
    body_index=0
    cur_line=body[body_index].split()

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                if len(cur_line)!=0:
                    array[x,y,z]=cur_line.pop(0)
                    #print("just loaded in value",array[x,y,z])
                else:
                    body_index+=1
                    cur_line=body[body_index].split()
                    array[x,y,z]=cur_line.pop(0)
                    #print("just loaded in value",array[x,y,z])
                    #print("Working on line",body_index)

    return array

