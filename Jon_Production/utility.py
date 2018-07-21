# Jon V

import numpy as np
import os
import subprocess
import json
import time
import copy
import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# ------------------------------------------------------
#         Quantum Espresso helper functions
# ------------------------------------------------------

# create supercell
def get_supercell(unit_cell, dim, unit_pos):
    # initialize position list
    positions = []

    # define bravais lattice vectors
    vec1 = np.array(unit_cell[0])
    vec2 = np.array(unit_cell[1])
    vec3 = np.array(unit_cell[2])

    # append positions of atoms in supercell
    for m in range(dim):
        for n in range(dim):
            for p in range(dim):
                for q in range(len(unit_pos)):
                    positions.append([unit_pos[q][0], \
                        list(np.array(unit_pos[q][1]) +\
                         m*vec1 + n*vec2 + p*vec3)])
                    
    # get supercell dimensions
    supercell = list(np.array(unit_cell)*dim)
                    
    return positions, supercell

# perturb the positions in a supercell
def perturb_struc(positions, pert_size):
    # loop through positions and add a random perturbation
    for n in range(len(positions)):
        for m in range(3):
            # get current coordinate
            coord_curr = positions[n][1][m]

            # get perturbation by drawing from uniform
            pert = np.random.uniform(-pert_size, pert_size)

            # perturb the coordinate
            positions[n][1][m] += pert
            
    return positions

# put supercell positions and cell parameters in QE friendly format
# based on Boris K's AP275 code
def get_position_txt(positions, supercell):
    
    # write atomic positions
    postxt = ''
    postxt += 'ATOMIC_POSITIONS {angstrom}'
    for pos in positions:
        postxt += '\n {} {:1.5f} {:1.5f} {:1.5f}'.format(pos[0], *pos[1])
        
    # write cell parameters
    celltxt = ''
    celltxt += 'CELL_PARAMETERS {angstrom}'
    for vector in supercell:
        celltxt += '\n {:1.5f} {:1.5f} {:1.5f}'.format(*vector)
    return postxt, celltxt

# get perturbed positions
def get_perturbed_pos(unit_cell, dim, unit_pos, pert_size):
    # get perturbed structure
    positions, supercell = get_supercell(unit_cell, dim, unit_pos)
    positions = perturb_struc(positions, pert_size)
    pos, cell = get_position_txt(positions, supercell)
    
    #get position array
    pos_array = [positions[n][1] for n in range(len(positions))]

    return pos, cell, pos_array, supercell, positions

# add atom labels to position array
def add_label(pos, pos_label):
    lab = []
    for n in range(len(pos)):
        lab.append([pos_label[n][0], pos[n]])
        
    return lab

# create text file
def write_file(fname, text):
    with open(fname, 'w') as fin:
        fin.write(text)

# run command
def run_command(command):
    myrun = subprocess.call(command, shell=True)

# create initial scf input
def get_scf_input(pos, cell, pseudo_dir, outdir, unit_cell, unit_pos, \
                ecut, nk, dim, nat, pert_size):
    
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
    """.format(pseudo_dir, outdir, \
               nat, ecut, cell, pos, nk)
    
    return scf_text

# run scf
def run_scf(scf_text, in_file, pw_loc, npool, out_file):

    # write input file
    write_file(in_file, scf_text)

    # call qe
    qe_command = 'mpirun {0} -npool {1} < {2} > {3}'.format(pw_loc, npool, in_file, out_file)
    run_command(qe_command)

# get forces in Ry/a.u.
# based on Steven T's MD parser
def parse_forces(outfile):
    # get lines
    with open(outfile, 'r') as outf:
        lines = outf.readlines()

    # use exclamation point to chop the file
    split_indexes=[N for N in range(len(lines)) if '!'==lines[N][0]]

    # cut out the first chunk 
    first_chunk=lines[0:split_indexes[0]]

    # carve the rest into chunks
    step_chunks = []
    for n in range(len(split_indexes)):
        step_chunks.append(lines[split_indexes[n]:split_indexes[n+1] \
                                 if n!=len(split_indexes)-1 else len(lines)])
        
    # loop through the chunks
    for current_chunk in step_chunks:
        # get force indices
        force_start_line = [line for line in current_chunk if 'Forces acting on atoms' in line][0]
        force_end_line   = [line for line in current_chunk if 'Total force' in line][0]
        force_start_index = current_chunk.index(force_start_line)+2
        force_end_index = current_chunk.index(force_end_line)-2

        # record forces
        forces = []
        for line in current_chunk[force_start_index:force_end_index+1]:
            forceline= line.split('=')[-1].split()
            forces.append([float(forceline[0]), \
                           float(forceline[1]), \
                           float(forceline[2])])

    return forces

# ------------------------------------------------------
#        molecular dynamics helper functions
# ------------------------------------------------------

# code adapted from Steven's MD engine

# assumptions for inputs:
# positions in angstrom
# forces in Ry/au
# mass in atomic mass units
# step in rydberg a.u.

# functions return updated positions in angstrom

# update first positions
def update_first(pos_curr, forces_curr, step, mass):
    # unit conversions
    force_conv = 25.71104309541616 * 1.602176620898e-19 / 1e-10 # Ry/a.u. to J/m
    mass_conv = 1.66053904020e-27 # atomic mass units to kg
    time_conv = 4.83776865301828e-17 # rydberg a.u. to s
    len_conv = 1e-10 # angstrom to meters
    
    # change in position
    change = ((1/2)*forces_curr*force_conv*(step*time_conv)**2/(mass*mass_conv))/len_conv
    
    # new positions
    pos_next = pos_curr + change
    
    return pos_next

# update positions using Verlet integration
def update_position(pos_curr, pos_prev, forces_curr, step, mass):
    # unit conversions
    force_conv = 25.71104309541616 * 1.602176620898e-19 / 1e-10 # Ry/a.u. to J/m
    mass_conv = 1.66053904020e-27 # atomic mass units to kg
    time_conv = 4.83776865301828e-17 # rydberg a.u. to s
    len_conv = 1e-10 # angstrom to meters
    
    # new positions
    pos_next = (2*pos_curr*len_conv -\
        pos_prev*len_conv +\
        forces_curr*force_conv*(step*time_conv)**2/(mass*mass_conv))/len_conv
    
    return pos_next

# ------------------------------------------------------
#              fingerprint helper functions
# ------------------------------------------------------

def get_cutoff_vecs(vec, brav_mat, brav_inv, vec1, vec2, vec3, cutoff):
    # get bravais coefficients
    coeff = np.matmul(brav_inv, vec)
    
    # get bravais coefficients for atoms within one super-super-super-cell
    coeffs = [[],[],[]]
    for n in range(3):
        coeffs[n].append(coeff[n])
        coeffs[n].append(coeff[n]-1)
        coeffs[n].append(coeff[n]+1)
        coeffs[n].append(coeff[n]-2)
        coeffs[n].append(coeff[n]+2)

    # get vectors within cutoff
    vecs = []
    dists = []
    for m in range(len(coeffs[0])):
        for n in range(len(coeffs[1])):
            for p in range(len(coeffs[2])):
                vec_curr = coeffs[0][m]*vec1 + coeffs[1][n]*vec2 + coeffs[2][p]*vec3
                
                dist = np.linalg.norm(vec_curr)

                if dist < cutoff:
                    vecs.append(vec_curr)
                    dists.append(dist)
                    
    return vecs, dists

# given a supercell and an atom number, return symmetry vectors
def symmetrize_forces(pos, atom, cutoff, eta_lower, eta_upper, eta_length, brav_mat, brav_inv,\
                     vec1, vec2, vec3):

    # set atom position
    pos_atom = np.array(pos[atom])
    etas = np.logspace(eta_lower, eta_upper, eta_length)

    # initialize symmetry vectors
    symm_x = np.zeros([len(etas)])
    symm_y = np.zeros([len(etas)])
    symm_z = np.zeros([len(etas)])

    # loop through positions to find all atoms and images in the neighborhood
    for n in range(len(pos)):
        # note that images of the atom don't contribute to symmetry vectors
        if n != atom:
            # position relative to reference atom
            diff_curr = np.array(pos[n]) - pos_atom

            # get images within cutoff
            vecs, dists = get_cutoff_vecs(diff_curr, brav_mat, \
                brav_inv, vec1, vec2, vec3, cutoff)

            # symmetrize according to Botu (2015)
            for vec, dist in zip(vecs, dists):
                # get cutoff factor
#                 cut_val = 0.5 * (np.cos(np.pi * dist / cutoff) + 1)
                cut_val = 1
                
                # get raw symmetry vectors
                symm_x += [(vec[0] / dist) * \
                        np.exp(-(dist / eta)**2) * cut_val for eta in etas]

                symm_y += [(vec[1] / dist) * \
                        np.exp(-(dist / eta)**2) * cut_val for eta in etas]

                symm_z += [(vec[2] / dist) * \
                        np.exp(-(dist / eta)**2) * cut_val for eta in etas]
    
    # concatenate the symmetry vectors to represent the full environment
    symm_x_cat = np.concatenate((symm_x, symm_y, symm_z))
    symm_y_cat = np.concatenate((symm_y, symm_z, symm_x))
    symm_z_cat = np.concatenate((symm_z, symm_x, symm_y))
                
    return symm_x_cat, symm_y_cat, symm_z_cat

# for a given supercell, calculate symmetry vectors for each atom
def augment_database(pos, forces, database, cutoff, eta_lower, eta_upper, eta_length, \
                            brav_mat, brav_inv, vec1, vec2, vec3):
    for n in range(len(pos)):
        # get symmetry vectors
        symm_x, symm_y, symm_z = symmetrize_forces(pos, n, cutoff, eta_lower, eta_upper, \
                                                   eta_length, brav_mat, brav_inv, vec1, vec2, vec3)
        
        # append symmetry vectors
        database['symms'].append(symm_x)
        database['symms'].append(symm_y)
        database['symms'].append(symm_z)
        
        # append force components
        database['forces'].append(forces[n][0])
        database['forces'].append(forces[n][1])
        database['forces'].append(forces[n][2])

# normalize the symmetry vectors in the training set
def normalize_symm(td):
    symm_len = len(td['symms'][0])
    td_size = len(td['symms'])

    # initialize normalized symmetry vector
    td['symm_norm'] = copy.deepcopy(td['symms'])
    
    # store normalization factors
    td['symm_facs'] = []

    for m in range(symm_len):  
        # calculate standard deviation of current symmetry element
        vec = np.array([td['symms'][n][m] for n in range(td_size)])
        vec_std = np.std(vec)
        
        # store standard deviation
        td['symm_facs'].append(vec_std)

        # normalize the current element
        for n in range(td_size):
            td['symm_norm'][n][m] = td['symm_norm'][n][m] / vec_std


# normalize forces
def normalize_force(td):
    td_size = len(td['forces'])

    # initialize normalized force vector
    td['forces_norm'] = copy.deepcopy(td['forces'])

    # calculate standard deviation of force components
    vec_std = np.std(td['forces'])

    # store standard deviation
    td['force_fac'] = vec_std

    # normalize the forces
    for n in range(td_size):
        td['forces_norm'][n] = td['forces_norm'][n] / vec_std


# augment and normalize
def aug_and_norm(pos, forces, database, cutoff, eta_lower, eta_upper, eta_length, \
                            brav_mat, brav_inv, vec1, vec2, vec3):
    # augment
    augment_database(pos, forces, database, cutoff, eta_lower, eta_upper, eta_length, \
                            brav_mat, brav_inv, vec1, vec2, vec3)
    
    # normalize forces and symmetry vectors
    normalize_force(database)
    normalize_symm(database)

# ------------------------------------------------------
#           Gaussian Process helper functions
# ------------------------------------------------------

# train GP model on the current database
# code adapted from Simon B
def train_gp(db, length_scale, length_scale_min, length_scale_max):
    # set kernel
    kernel = RBF(length_scale=length_scale, length_scale_bounds=(length_scale_min, length_scale_max))

    # make GP
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    
    # fit GP model
    x_train = db['symm_norm']
    y_train = db['forces_norm']
    gp.fit(x_train, y_train)
    
    return gp

# predict force with the current gp model
def gp_pred(symm, norm_fac, gp):
    # predict with model
    pred = gp.predict(symm.reshape(1,-1), return_std = True)
    force_pred = pred[0][0] * norm_fac
    std_pred = pred[1][0] * norm_fac
    
    return force_pred, std_pred

# predict force component p for a given atom
def pred_comp(pos_curr, atom, cutoff, eta_lower, eta_upper, eta_length, brav_mat, brav_inv,\
             vec1, vec2, vec3, p, db, gp, force_conv):
    
    # symmetrize chemical environment
    symm = symmetrize_forces(pos_curr, atom, cutoff, eta_lower, eta_upper, eta_length, brav_mat, brav_inv,\
             vec1, vec2, vec3)

    symm_comp = symm[p]
    symm_norm = np.array([symm_comp[q] / db['symm_facs'][q] for q in range(len(symm_comp))])

    # estimate the force component and model error
    norm_fac = db['force_fac']
    force_pred, std_pred = gp_pred(symm_norm, norm_fac, gp)

    # calculate error
    err_pred = std_pred * force_conv

    return force_pred, err_pred



# ------------------------------------------------------
#              Math helper functions
# ------------------------------------------------------


def first_derivative_2nd(fm, fp, h):
    """
    Computes the second-order accurate finite difference form of the first derivative
    which is (  fp/2 - fm/2)/(h)
    as seen on Wikipedia: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """
    if h == 0:
        print("Warning... Trying to divide by zero. Derivative will diverge.")
        return np.nan

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



# ------------------------------------------------------
#           Output file helper functions
# ------------------------------------------------------
def update_init():
    init_text = """Welcome to PyFly Version 0.0.
Authors: Jonathan Vandermause, Steven B. Torrisi, Simon Batzner, Alexie Kolpak, and \
Boris Kozinsky.
Timestamp: %s.\n \n""" % str(datetime.datetime.now())

    return init_text

