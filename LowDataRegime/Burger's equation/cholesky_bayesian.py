#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:44:06 2023

@author: matthieudarcy
"""

from scipy import io
import numpy as np

import matplotlib.pyplot as plt


from sklearn.gaussian_process import GaussianProcessRegressor
plt.style.use("seaborn")

from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel


from scipy.linalg import cholesky, cho_factor, cho_solve

import scipy

#%%

def get_data(ntrain, ntest):
    sub_x = 2 ** 6
    sub_y = 2 ** 6

    # Data is of the shape (number of samples = 2048, grid size = 2^13)
    data = io.loadmat("burgers_data_R10.mat")
    x_data = data["a"][:, ::sub_x].astype(np.float64)
    y_data = data["u"][:, ::sub_y].astype(np.float64)
    x_branch_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_branch_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]
    
        
    s = 2 ** 13 // sub_y  # total grid size divided by the subsampling rate
    grid = np.linspace(0, 1, num=2 ** 13)[::sub_y, None]
    
    return x_branch_train, y_train, x_branch_test, y_test, grid


    x_train = (x_branch_train, grid)
    x_test = (x_branch_test, grid)
    return x_train, y_train, x_test, y_test


x, y, x_test, y_test, grid = get_data(1000, 200)


idx = 2

plt.figure()
plt.plot(grid, x[idx])
plt.xlabel(r'$x$', size= 15)
plt.ylabel(r'$u_0(x)$', size= 15)

#%%

def compute_cho(kernel_u, kernel_v, grid):
    K = kernel_u(grid)
    G = kernel_v(grid)
    #print(np.linalg.cond(K))
    
    print(kernel_u, kernel_v)

    tau = 1e-8
    L_K = cholesky(K + tau*np.eye(K.shape[0]), lower=True)
    L_G = cholesky(G+ tau*np.eye(K.shape[0]), lower = False)
    
    return L_K, L_G

def precondition(L_K, L_G, u ,v):
    tau = 1e-8
    L_G_inv = np.linalg.inv(L_G + tau*np.eye(L_G.shape[0]))    
    return (L_K.T @ u[:, :, None]).squeeze(-1), (L_G_inv.T @ v[:, :, None]).squeeze(-1)
    
def compute_error(prediction, target):
    e = np.mean(np.linalg.norm(prediction - target, axis = -1)/np.linalg.norm(target, axis = -1))
    
    return e

def optimal_recovery(K, G, L_K, L_G, u, v):
    tau = 0
    #L_K_inv = np.linalg.inv(L_K + tau*np.eye(K.shape[0]))
    L_G_inv = np.linalg.inv(L_G + tau*np.eye(K.shape[0]))
    
    u = np.linalg.solve(L_K.T, u[:, :, None]).squeeze(-1)
    #u_recov = np.squeeze(K@scipy.linalg.cho_solve((L_K, True), u[None] ))
    u_recov = np.squeeze(K@scipy.linalg.solve(K, u.T, assume_a = 'pos' )).T
    v_recov = np.squeeze(G@L_G_inv@v[:, :, None])
    
    return u_recov, v_recov

def train_test(x_train, x_test, y_train, y_test):
    kernel = Matern(nu = 2.5, length_scale = 1.0)
    gp = GaussianProcessRegressor(kernel, alpha = 1e-10,  normalize_y = False, random_state= 6032023) 
    
    gp.fit(x_train, y_train)
    pred= gp.predict(x_test)
    #pred_train = gp.predict(x_train)

    #e = compute_error_dataset(y_test, pred, knots, k)

    return pred, gp
 

#%%   


kernel_u =Matern(nu = 1.5, length_scale = 0.1) 
kernel_v = Matern(nu = 1.5, length_scale = 0.1)
L_K, L_G = compute_cho(kernel_u, kernel_v, grid)

x_train, y_train = precondition(L_K, L_G, x, y)
x_val, y_val = precondition(L_K, L_G, x_test, y_test)


pred, gp = train_test(x_train, x_val, y_train, y_val)
pred_train = gp.predict(x_train)

e = compute_error(pred, y_val)
e_train = compute_error(pred_train, y_train)
print(e, e_train)
#%%

# pointwise prediction
K = kernel_u(grid)
G = kernel_v(grid)
_, pred_train_point = optimal_recovery(K, G, L_K, L_G, x_train, pred_train)
_, pred_point = optimal_recovery(K, G, L_K, L_G, x_val, pred)

e = compute_error(pred_point, y_test)
e_train = compute_error(pred_train_point, y)
print(e, e_train)


#%% Bayesian optimization 

from skopt import gp_minimize
from sklearn.model_selection import KFold

def f_cv(x, y, nu_u, l_u, nu_v, l_v):
    
    kernel_u =Matern(nu = nu_u, length_scale = l_u) 
    kernel_v = Matern(nu = nu_v, length_scale = l_v)
    
    print(kernel_u, kernel_v)
    
    kf = KFold(n_splits=5)
    
    L_K, L_G = compute_cho(kernel_u, kernel_v, grid)
    
    e = 0
    for train_index, test_index in kf.split(x):
        #print("TRAIN:", train_index, "TEST:", test_index)
        
        x_train, y_train = precondition(L_K, L_G, x[train_index], y[train_index])
        x_val, y_val = precondition(L_K, L_G, x[test_index], y[test_index])
        
        pred, gp = train_test(x_train, x_val, y_train, y_val)
        
        e += compute_error(pred, y_val)
    #print(e)
    return (e/5).item()


f_opt = lambda param: f_cv(x, y, param[0], param[1], param[2], param[3])

#%%


from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

space= [ Categorical([0.5, 1.5, 2.5, np.inf], name='nu_u'),
        Real(0.00001, 100000.0, name='length_scale_u'),
        Categorical([0.5, 1.5, 2.5, np.inf], name='nu_v'),
        Real(0.00001, 100000.0, name='length_scale_v')
         ]

x0 = [1.5, 0.1, 1.5, 0.1]


#x0 = [1.5, 10000, 1.5, 0.1]

#%%

print(f_opt(x0))