#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:56:25 2018

@author: jonpvandermause
"""

from numpy.linalg import inv
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from KRR_reproduce import *
from scipy.interpolate import interp1d
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

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
    kern = np.exp(-dist**2/(2*sig**2))
    
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

def get_dist(pot,pots,sig):
    dists = []
    for n in range(len(pots)):
        pot_curr = pots[n]
        dist = get_kern(pot, pot_curr, sig)
        dists.append(dist)
        
    return dists
        

def pred_fast(dist,beta):
    pred = 0
    
    for n in range(len(dist)):
        pred += dist[n]*beta[n]
        
    return pred
            


rng = np.random.RandomState(0)

def get_train_test(M, N, input_dim, pos, ens, fours):
    # define training and test indices
    train_indices = [int(n) for n in np.round(np.linspace(0,N-1,M))]
    test_indices = [n for n in range(N) if n not in train_indices]

    # define train and test sets
    train_set = np.reshape(np.array([pos[n] for n in train_indices]),(M,input_dim))
    test_set = np.reshape(np.array([pos[n] for n in test_indices]),(N-M,input_dim))

    train_ens = np.reshape(np.array([ens[n] for n in train_indices]),(M,1))
    test_ens = np.reshape(np.array([ens[n] for n in test_indices]),(N-M,1))

    train_fours = np.array([fours[n] for n in train_indices])
    test_fours = np.array([fours[n] for n in test_indices])
    
    return  train_set, test_set, train_ens, test_ens, train_fours, test_fours

def fit_krr(train_X, train_Y, test_X, test_Y, alphas, gammas, cv):
    # define model
    kr = GridSearchCV(KernelRidge(kernel='rbf'), cv=cv,
                      param_grid={"alpha": alphas,
                                  "gamma": gammas})
    
    # fit model
    kr.fit(train_X, train_Y)
    
    # predict test energies
    y_kr = kr.predict(test_X)
    
    # calculate MAE and max error
    errs = np.abs(test_Y-y_kr)
    MAE = np.mean(np.abs(test_Y-y_kr))
    max_err = np.max(np.abs(test_Y-y_kr))
    
    return kr, y_kr, errs, MAE, max_err

def fit_quick(train_X, train_Y, alpha, gamma):
    kr = KernelRidge(kernel='rbf',alpha = alpha, gamma = gamma)
    kr.fit(train_X, train_Y)
    return kr

def fit_KS(train_set, train_ens, test_set, test_ens, alphas, gammas, cv):
    [kr, y_kr, errs, MAE, max_err] = fit_krr(train_set, \
                                         train_ens, test_set, test_ens, \
                                         alphas, gammas, cv)
    
    return kr, y_kr, errs, MAE, max_err

def fit_KS_pot(train_set, train_ens, test_set, test_ens, alphas, gammas, cv, seps):
    [kr, y_kr, errs, MAE, max_err] = fit_krr(train_set, \
                                         train_ens, test_set, test_ens, \
                                         alphas, gammas, cv)
    
    # check MAE in kcal/mol
    print(kr.best_estimator_)
    
    return kr, y_kr, errs, MAE, max_err

def spline_test(train_set, train_ens, test_set, test_ens):
    f = interp1d(train_set.reshape(len(train_set),), \
                  train_ens.reshape(len(train_ens),), kind='cubic')
    
    err = np.mean(np.abs(f(test_set)-test_ens))
    
    return err

def get_potentials(train_set, M, grid_len = 5.29177*2):
    # get potential kernel
    pots = []
    for n in range(M):
        dist = train_set[n]
        pot = pot_rep(dist, grid_len)
        pots.append(pot) 
        
    pots = np.array(pots)
    pots = np.reshape(pots,(M, pots.shape[1]**3))
    
    return pots

def pos_to_four(train_set, train_fours, M, alpha, gamma, input_dim, comp):
    # build position to Fourier models
    krs = []
    no_four = comp
    train_X = np.reshape(train_set, (M,input_dim))

    for i in range(no_four):
        four1 = i
        for j in range(no_four):
            four2 = j
            for k in range(no_four):
                four3 = k
                
                # build model
                train_Y = np.reshape(train_fours[:,four1, four2, four3],(M,1))
                kr = fit_quick(train_X, train_Y, alpha, gamma)

                krs.append(kr)    
    return krs

def four_to_en(train_fours, train_ens, test_fours, test_ens, M, N, alphas, gammas, comp):
    # build Fourier to energy model
    train_X = np.reshape(train_fours,(M,comp**3))
    test_X = np.reshape(test_fours,(N-M,comp**3))
    train_Y = train_ens
    
    if M < 10:
        cv = M - 1
    else:
        cv = 9

    [FE_kr, y_kr, errs, MAE, max_err] = fit_krr(train_X, \
                                             train_Y, test_X, test_ens, \
                                             alphas, gammas, cv)
    
    return FE_kr, y_kr, errs, MAE, max_err

def four_to_en_full(train_fours, train_ens, test_fours, test_ens, M, N, alphas, gammas):
    # build Fourier to energy model
    train_X = train_fours
    test_X = test_fours
    train_Y = train_ens
    
    if M < 10:
        cv = M - 1
    else:
        cv = 9

    [FE_kr, y_kr, errs, MAE, max_err] = fit_krr(train_X, \
                                             train_Y, test_X, test_ens, \
                                             alphas, gammas, cv)
    
    return FE_kr, y_kr, errs, MAE, max_err

def doub_map_fast(test_set, test_ens, krs, FE_kr, M, N, comp):
    # perform double mapping
    no_four = comp

    # get density
    test_dens = np.zeros([N-M, no_four, no_four, no_four])
    count = 0
    for i in range(no_four):
        for j in range(no_four):
            for k in range(no_four):
                # use model to calculate density
                kr_curr = krs[count]
                test_dens[:,i,j,k]=kr_curr.predict(test_set).reshape(-1,)

                count+=1

    # get energy
    pred_ens = FE_kr.predict(np.reshape(test_dens,(N-M,comp**3)))
    HK_errs = test_ens-pred_ens
    MAE = np.mean(np.abs(HK_errs))
    max_err = np.max(np.abs(HK_errs))
    
    return HK_errs, MAE, max_err
            
        
    