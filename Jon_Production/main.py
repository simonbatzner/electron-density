# Jon V

import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from utility import *

# ------------------------------------------------------
#          set parameters for on-the-fly run
# ------------------------------------------------------

# MD parameters
frames = 50 # number of MD frames
noa = 54 # number of atoms in 3x3x3 supercell
step = 20 # step size in rydberg a.u.
mass = 28.0855 # Si mass in atomic mass units

# crystal structure
alat = 5.431
unit_cell = [[0.0, alat/2, alat/2], [alat/2, 0.0, alat/2], \
					[alat/2, alat/2, 0.0]] # fcc primitive cell
unit_pos = [['Si',[0,0,0]],['Si',[alat/4, alat/4, alat/4]]]
brav_mat = np.array([[0.0, alat/2, alat/2], [alat/2, 0.0, alat/2], \
					[alat/2, alat/2, 0.0]])*3
brav_inv = np.linalg.inv(brav_mat)

# bravais vectors
vec1 = brav_mat[:,0]
vec2 = brav_mat[:,1]
vec3 = brav_mat[:,2]

# GP parameters
length_scale = 1
length_scale_min = 1e-5
length_scale_max = 1e5

# threshold parameters
force_conv = 25.71104309541616 # Ry/au to eV/A
err_thresh = 0.09 # in eV/A

# fingerprint parameters
eta_lower = 0
eta_upper = 2
eta_length = 10
cutoff = 8 # in angstrom

# QE parameters
ecut = 18.0 # plane wave cutoff energy
nk = 4 # size of kpoint grid
dim = 3 # size of supercell
nat = 2 * dim**3 # number of atoms in supercell
pert_size = 0.05 * alat # size of initial perturbation
npool = 36 # number of k-pt pools

# QE locations
pseudo_dir = '/n/home03/jonpvandermause/qe-6.2.1/pseudo'
outdir='/n/home03/jonpvandermause/Production'
pw_loc = '/n/home03/jonpvandermause/qe-6.2.1/bin/pw.x'
in_file = 'first_scf.in'
out_file = 'first_scf.out'

# ------------------------------------------------------
#             perform on-the-fly run
# ------------------------------------------------------

# initialize the training set
db = {'symms':[], 'forces':[]}
pos_store = []
force_store = []
all_err_preds = []

# loop through MD snapshots
for n in range(frames):
    
    print('frame no '+str(n))
    
    # if first snapshot, run DFT and add to training set
    if n == 0:
        # get initial positions
        pos_text, cell, pos, supercell, pos_label = \
            get_perturbed_pos(unit_cell, dim, unit_pos, pert_size)

        pos_store.append(pos)

    	# call QE
        scf_text = get_scf_input(pos_text, cell, pseudo_dir, outdir, unit_cell, unit_pos, \
                ecut, nk, dim, nat, pert_size)

        run_scf(scf_text, in_file, pw_loc, npool, out_file)

        # parse output file
        forces = parse_forces(out_file)
        force_store.append(forces)

        # augment and normalize the database
        aug_and_norm(pos, forces, db, cutoff, eta_lower, eta_upper, eta_length, \
                            brav_mat, brav_inv, vec1, vec2, vec3)
        
        # train GP model
        gp = train_gp(db, length_scale, length_scale_min, length_scale_max)

        # update positions
        pos_prev = np.array(pos)
        pos_curr = update_first(np.array(pos), np.array(forces), step, mass)
        pos_store.append(pos_curr)
        
    else:
        # make GP force predictions for each atom (this should be parallelized)
        forces_curr = []
        tot_force = []
        err_preds = []
        thresh_count = 0
        thresh = 0

        for m in range(noa):
            # symmetrize atomic environment
            atom = m
            symm = symmetrize_forces(pos_curr, atom, cutoff, eta_lower, eta_upper, eta_length, brav_mat, brav_inv,\
                     vec1, vec2, vec3)

            forces_curr.append([])
            
            # loop over three symmetry vectors
            for p in range(3):
                symm_comp = symm[p]
                symm_norm = np.array([symm_comp[q] / db['symm_facs'][q] for q in range(len(symm_comp))])
                
                # estimate the force component and model error
                norm_fac = db['force_fac']
                force_pred, std_pred = gp_pred(symm_norm, norm_fac, gp)
                
                # calculate error
                err_pred = std_pred * force_conv

                # store forces and error
                forces_curr[m].append(force_pred)
                tot_force.append(np.abs(force_pred * force_conv))
                all_err_preds.append(err_pred)
                err_preds.append(err_pred)
        
        # get average predicted error
        avg_err = np.mean(err_preds) 
        print('average predicted error is '+str(avg_err))   

        # if average error is above the threshold, call QE and retrain
        if avg_err > err_thresh:
            print('above threshold. calling QE')

            # call QE 
            pos_label = add_label(pos_curr, pos_label)
            pos_text, cell_text = get_position_txt(pos_label, supercell)
            scf_text = get_scf_input(pos_text, cell_text, pseudo_dir, outdir, unit_cell, unit_pos, \
                ecut, nk, dim, nat, pert_size)
            run_scf(scf_text, in_file, pw_loc, npool, out_file)
            forces_curr = parse_forces(out_file)
            force_store.append(forces_curr)

            # augment and normalize the database
            aug_and_norm(pos_curr, forces, db, cutoff, eta_lower, eta_upper, eta_length, \
                                brav_mat, brav_inv, vec1, vec2, vec3)
            
            # train GP model
            gp = train_gp(db, length_scale, length_scale_min, length_scale_max)

            # update positions
            pos_new = update_position(np.array(pos_curr), np.array(pos_prev), np.array(forces_curr), step, mass)
            pos_prev = pos_curr
            pos_curr = pos_new
            pos_store.append(pos_curr)
            
        # if average error is below the threshold, update positions using predicted forces
        else:
            print('below threshold. using GP forces')

            # store forces and update positions
            pos_new = update_position(np.array(pos_curr), np.array(pos_prev), np.array(forces_curr), step, mass)
            pos_prev = pos_curr
            pos_curr = pos_new
            force_store.append(forces_curr)
            pos_store.append(pos_curr)

        # print results before moving onto the next snapshot
        print('database size is '+str(len(db['symms'])))
        print('average predicted error is '+str(np.mean(np.array(err_preds))) + 'eV/A.')
        print('average of all predicted errors is '+\
            str(np.mean(np.array(all_err_preds)))+' eV/A.')
        print('average force is '+str(np.mean(np.array(tot_force)))+' eV/A.')
        
        