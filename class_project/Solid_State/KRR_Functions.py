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