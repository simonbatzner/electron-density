#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 17:20:12 2018

@author: jonpvandermause
"""

from Aluminum_Workflow import *

# choose save directory
save_dir = os.environ['WORKDIR']+'/Al_Data/Store/'

# run MD
timestep= 0.01
temperature = 300
size = 1
nsteps = 4000
tout = 20

output, rdfs = compute_dynamics_NVT(size=size, \
            timestep=timestep, nsteps=nsteps,\
            temperature=temperature, tout=tout)

[simtime, pe, ke, energy, temp, press, dens, msd] = output

# save simulation times and energies
np.save(save_dir+'MD_simtimes',simtime)
np.save(save_dir+'MD_energies',energy)

# set the dumpfile
dumpfile = os.environ['WORKDIR'] + '/Al_Data/LAMMPS/size_' + str(size)+'/dump.atom'

# create dictionary of positions
noa = 4 * size**3
nop = int(nsteps/tout + 1)
pos = {}

with open(dumpfile) as f:
    for line in f:
        # split the line
        numbers_str = line.split()
        
        # store position vectors in a dictionary
        if len(numbers_str) == 5:
            atom_label = numbers_str[0]
            pos_list = [float(x) for x in numbers_str]
            
            if len(pos) != noa:
                pos[atom_label] = [pos_list[2:5]]
            else:
                pos[atom_label].append(pos_list[2:5])

# convert lists to np arrays
for n in pos:
    pos[n] = np.array(pos[n])
    
# create np array of positions
pos_array = np.zeros([noa, 3, nop])

for m in range(noa):
    at = m+1
    for n in range(nop):
        pos_array[m,:,n] = pos[str(at)][n,:]
        
# store positions
np.save(save_dir+'MD_positions',pos_array)

# set Al parameters
dirnums = np.arange(0,201,1)
nk = 15
ecut = 40
size = 1
ncpu = 24
alat = 4.1

str_pref = os.environ['WORKDIR']+'/Al_Data/Store/'

# loop through directory numbers
for n in range(len(dirnums)):
    # choose directory number
    dir_num = dirnums[n]
    
    # define positions
    pos_curr = pos_array[:,:,dir_num]
    np.save(str_pref+'pos_store/pos'+str(n),pos_curr)

    # create aluminum structure
    Al_struc = struc_cust(size, alat, pos_array, dir_num)

    # calculate the energy
    en = compute_energy(alat, nk, ecut, Al_struc, ncpu)
    energy = en['energy']

    np.save(str_pref+'en_store/energy'+str(n),energy)

    # calculate the density
    quick_density_gen()

    # parse the density file
    dirname = 'Al_Files'
    file_path = os.environ['WORKDIR']+'/Al_Data/'+dirname+'/Al.rho.dat'
    dens = rho_to_numpy(file_path,natoms=4)

    # take the fourier transform
    four = sp.fftpack.fftn(dens)

    # form grid of 25x25x25 fourier coefficients
    four_store = four[0:25, 0:25, 0:25]      
    np.save(str_pref+'four_store/four'+str(n),four_store)
    
    print(energy)
    print(n)