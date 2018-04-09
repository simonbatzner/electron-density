#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:56:25 2018

@author: jonpvandermause
"""

import numpy as np
from numpy.linalg import inv

# represent potential as an artificial gaussian
# takes spacing between H atoms (angstrom), grid length (angstrom),
# and grid size as input and returns 3d gaussian potential
# following Brockherde (2017), set 0.08 as default grid spacing
def pot_rep(dist, grid_len, grid_space=0.08):
    # x coordinates of hydrogen atoms
    pos1 = -dist / 2
    pos2 = dist / 2
    
    # grid edges
    edge1 = -grid_len / 2
    edge2 = grid_len / 2
    
    # define grid
    x = np.arange(edge1, edge2, grid_space)
    y = np.arange(edge1, edge2, grid_space)
    z = np.arange(edge1, edge2, grid_space)
    xv, yv, zv = np.meshgrid(x, y, z)
    
    # following Brockherde (2017), use gam = 0.2 A and a coarse grid with
    # grid spacing del = 0.08 (see "Molecules" section)
    gam = 0.2
    
    # define distances between grid points and atoms
    dist1 = (xv-pos1)**2+yv**2+zv**2
    dist2 = (xv-pos2)**2+yv**2+zv**2
    gauss = np.exp(-dist1 / (2*gam**2))+np.exp(-dist2 / (2*gam**2))
    
    return gauss

# get the kernel between two gaussian potentials
# sigma chosen by cross validation
def get_kern(v1, v2, sig):
    # get distance between two potentials by taking the Frobenius norm
    dist = np.linalg.norm(v1-v2)
    
    # use gaussian kernel
    kern = np.exp(-dist/(2*sig**2))
    
    return kern

# given a set of potentials, construct kernel matrix
# assume potentials are given as a list of numpy arrays
def get_kern_mat(pots, sig):
    no_pots = len(pots)
    kern_mat = np.empty([no_pots, no_pots])
    
    for m in range(no_pots):
        pot_m = pots[m]
        
        for n in range(no_pots):
            pot_n = pots[n]
            
            kern_mat[m, n]=get_kern(pot_m, pot_n, sig)
            
    return kern_mat
            
# calculate optimal KRR model
def get_beta(kern_mat, data_vec, lam):
    dim = kern_mat.shape[0]
    in_mat = kern_mat + lam * np.eye(dim)
    inv_mat = inv(in_mat)
    beta = np.matmul(inv_mat, data_vec)
    
    return beta

# predict fourier coefficient
def pred_four(pot,pots,beta,sig):
    four_pred = 0
    
    for n in range(len(pots)):
        pot_curr = pots[n]
        dist = get_kern(pot, pot_curr, sig)
        four_pred = four_pred + dist * beta[n]
        
    return four_pred
            
            
        
    