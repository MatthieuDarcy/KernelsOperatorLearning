#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:26:42 2023

@author: matthieudarcy
"""

from scipy import io
import numpy as np

import matplotlib.pyplot as plt

plt.style.use("seaborn")

from sklearn.decomposition import PCA

from sklearn.linear_model import Ridge, LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from scipy.linalg import cholesky, cho_factor, cho_solve


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF




from math import inf


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


#%% No Cholesky


def train_test(x_train, x_test, y_train, y_test):
    kernel = Matern(nu = 2.5, length_scale = 1.0)
    gp = GaussianProcessRegressor(kernel, alpha = 1e-10,  normalize_y = True, random_state= 6032023) 
    
    gp.fit(x_train, y_train)
    pred= gp.predict(x_test)
    #pred_train = gp.predict(x_train)

    #e = compute_error_dataset(y_test, pred, knots, k)

    return pred, gp

pred, GP = train_test(x, x_test, y, y_test)
pred_train = GP.predict(x)
e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y, axis = -1)/np.linalg.norm(y, axis = -1))


print(e, e_train)
idx = 15
i = 0

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))
ax1.plot(grid, y[idx], label = "True")
ax1.plot(grid, pred_train[idx], label = "Prediction")
ax1.set_xlabel(r'$x$', size= 15)
ax1.set_ylabel(r'$u(x,1)$', size= 15)
ax1.set_title("Prediction on the training set", size = 15)
ax1.legend()




#%% Choleksy transform

# kernel_u = Matern(nu = 0.5, length_scale = 0.1) 
# kernel_v = Matern(nu = 0.5, length_scale = 0.1)

# K = kernel_u.__call__(grid)
# G = kernel_v.__call__(grid)

# L_K = cholesky(K, lower=False)
# L_G = cholesky(G, lower = False)

# tau = 1e-8
# L_K_inv = np.linalg.inv(L_K + tau*np.eye(K.shape[0]))
# L_G_inv = np.linalg.inv(L_G + tau*np.eye(K.shape[0]))


# #%% Cholesky transformation

# x_train = []
# for i in range(x.shape[0]):
#     x_train.append(L_K_inv.T@x[i])
# x_train = np.array(x_train)

# y_train = []
# for i in range(y.shape[0]):
#     y_train.append(L_G_inv.T@y[i])
# y_train = np.array(y_train)


# x_val = []
# for i in range(x_test.shape[0]):
#     x_val.append(L_K_inv.T@x_test[i])
# x_val = np.array(x_val)

# y_val = []
# for i in range(y_test.shape[0]):
#     y_val.append(L_G_inv.T@y_test[i])
# y_val = np.array(y_val)


#%% Choelsky precondition

kernel_u = Matern(nu = 0.5, length_scale = 0.1) 
kernel_v = Matern(nu = 0.5, length_scale = 0.1)
def cholesky_transform(kernel_u, kernel_v, grid, u, v):
    K = kernel_u(grid)
    G = kernel_v(grid)

    L_K = cholesky(K, lower=False)
    L_G = cholesky(G, lower = False)

    tau = 1e-8
    L_K_inv = np.linalg.inv(L_K + tau*np.eye(K.shape[0]))
    L_G_inv = np.linalg.inv(L_G + tau*np.eye(K.shape[0]))

    
    return (L_K_inv.T @ u[:, :, None]).squeeze(-1), (L_G_inv.T @ v[:, :, None]).squeeze(-1)

x_train, y_train = cholesky_transform(kernel_u, kernel_v, grid, x, y)
x_val, y_val = cholesky_transform(kernel_u, kernel_v, grid, x_test, y_test)


#%% Optimal recovery: pointwize values

def optimal_recovery(kernel_u, kernel_v, grid, u, v):
    K = kernel_u(grid)
    G = kernel_v(grid)

    L_K = cholesky(K, lower=False)
    L_G = cholesky(G, lower = False)

    tau = 1e-8
    L_K_inv = np.linalg.inv(L_K + tau*np.eye(K.shape[0]))
    L_G_inv = np.linalg.inv(L_G + tau*np.eye(K.shape[0]))
    
    u_recov = np.squeeze(K@L_K_inv@u[:, :, None])
    v_recov = np.squeeze(G@L_G_inv@v[:, :, None])
    
    return u_recov, v_recov

u_recov, v_recov = optimal_recovery(kernel_u, kernel_v, grid, x_train, y_train)
    
e_u = np.mean(np.linalg.norm(u_recov - x, axis = -1)/np.linalg.norm(x, axis = -1))
e_v = np.mean(np.linalg.norm(v_recov - y, axis = -1)/np.linalg.norm(y, axis = -1))

print(e_u, e_v)


u_recov, v_recov = optimal_recovery(kernel_u, kernel_v, grid, x_val, y_val)
    
e_u = np.mean(np.linalg.norm(u_recov - x_test, axis = -1)/np.linalg.norm(x_test, axis = -1))
e_v = np.mean(np.linalg.norm(v_recov - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(e_u, e_v)
#%% Optimal recovery: pointwise values

# u_recov = []
# for i in range(x.shape[0]):
#     u_recov.append(K@L_K_inv@x_train[i])
# u_recov = np.array(u_recov)

    
# v_recov = []
# for i in range(y.shape[0]):
#     v_recov.append(G@L_G_inv@pred_train[i])
# v_recov = np.array(v_recov)

# e_u = np.mean(np.linalg.norm(u_recov - x, axis = -1)/np.linalg.norm(x, axis = -1))
# e_v = np.mean(np.linalg.norm(v_recov - y, axis = -1)/np.linalg.norm(y, axis = -1))

# print(e_u, e_v)

# u_recov = []
# for i in range(x_test.shape[0]):
#     u_recov.append(K@L_K_inv@x_val[i])
# u_recov = np.array(u_recov)

    
# v_recov = []
# for i in range(y_test.shape[0]):
#     v_recov.append(G@L_G_inv@y_val[i])
# v_recov = np.array(v_recov)

# e_u = np.mean(np.linalg.norm(u_recov - x_test, axis = -1)/np.linalg.norm(x_test, axis = -1))
# e_v = np.mean(np.linalg.norm(v_recov - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

# print(e_u, e_v)


#%%

pred, gp = train_test(x, x_test, y_train, y_val)
pred_train = gp.predict(x)
e = np.mean(np.linalg.norm(pred - y_val, axis = -1)/np.linalg.norm(y_val, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y_train, axis = -1)/np.linalg.norm(y_train, axis = -1))

print(e, e_train)

print(gp.kernel_)

#%% Recovering the pointwise measurements




# pred_point = []
# for i in range(y_test.shape[0]):
#     pred_point.append(G@L_G_inv@pred[i])
    
# pred_point = np.array(pred_point)
    
# pred_point_train = []
# for i in range(x.shape[0]):
#     pred_point_train.append(G@L_G_inv@pred_train[i])
# pred_point_train = np.array(pred_point_train)

pred_point_train, pred_point = optimal_recovery(kernel_u, kernel_v, grid, pred_train, pred)

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print(e, e_train)

idx = 15

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))
ax1.plot(grid, y[idx], label = "True")
ax1.plot(grid, pred_point_train[idx], label = "Prediction")
ax1.set_xlabel(r'$x$', size= 15)
ax1.set_ylabel(r'$u(x,1)$', size= 15)
ax1.set_title("Prediction on the training set", size = 15)
ax1.legend()

ax2.plot(grid, y_test[idx],  label = "True")
ax2.plot(grid, pred_point[idx],  label = "Prediction")
ax2.set_xlabel(r'$x$', size= 15)
ax2.set_ylabel(r'$u(x, 1)$', size= 15)
ax2.set_title("Prediction on the test set", size = 15)
ax2.legend()

#%% optimizing the K, G parameters
# kernel_u = Matern(nu = 0.5, length_scale = 1.0) 
# gp_u = GaussianProcessRegressor(kernel_u, alpha = 1e-10,  normalize_y = False, random_state= 6032023, optimizer = None) 
# gp_u.fit(grid, x[0])


# theta_test = np.array([1.0])
# #gp_u.theta = theta_te1t

# #K, grad_k = gp_u.log_marginal_likelihood(np.array([10.0]),eval_gradient=True, clone_kernel=False)

# K, grad_k = kernel_u(grid, eval_gradient = True)



# def log_marginal_likelihood(y, gp, theta):
#     values = []
#     #print(theta)
#     for sample in y:
#         #print(sample[50])
#         gp.y_train_ = sample
#         #gp.fit(grid,sample)
#         #print(sample[0])
#         #print(sample.shape)
#         #print(gp.L_)
#         #print(gp.alpha_)
#         values.append(gp.log_marginal_likelihood(theta = theta,  eval_gradient = True))
        
    
#     return np.mean(np.array(values, dtype = object), axis = 0)
    

# values =log_marginal_likelihood(x, gp_u, theta_test) 
# #print()

# #%%


# def obj_func(theta, gp, y):
    
#     value, grad = log_marginal_likelihood(y, gp, theta)
#     return -value, -grad

# initial_theta = np.array([1.0])
# print(obj_func(initial_theta, gp_u, x))
# #%%
# bnds =  ((1e-5, 1e5))
# opt_res = scipy.optimize.minimize(
#                 obj_func,
#                 initial_theta,
#                 args = (gp_u, x),
#                 method="BFGS",
#                 jac=True,
#                 options = {"disp": True})


# #%%

# print(opt_res.x)

#%% Choosing the parmaters using MLE

def log_marginal_likelihood(y, gp, theta):
    values = []
    for sample in y:
        gp.y_train_ = sample
        values.append(gp.log_marginal_likelihood(theta = theta,  eval_gradient = True))
        
    
    return np.mean(np.array(values, dtype = object), axis = 0)

def mle(grid, function_samples, kernel, theta_init, bnds):
    gp = GaussianProcessRegressor(kernel, alpha = 1e-10,  normalize_y = False, optimizer = None) 
    gp.fit(grid, function_samples[0])
    
    
    def obj_func(theta, gp, y):
        
        value, grad = log_marginal_likelihood(y, gp, theta)
        return -value, -grad
    
    opt_res = scipy.optimize.minimize(
                    obj_func,
                    theta_init,
                    args = (gp, function_samples),
                    method="L-BFGS-B",
                    jac=True,
                    bounds= bnds,
                    options = {"disp": False})
    print(opt_res.message)
    return opt_res.x

#%%
theta_init = np.array([1.0])
bnds =  ((1e-5, 1e5),)
kernel_u = Matern(nu = 0.5, length_scale = 1.0)
param_u = mle(grid, x, kernel_u, theta_init, bnds)

#%%
theta_init = np.array([1.0])
bnds =  ((1e-10, 1e5),)
kernel_v = Matern(nu = 0.5, length_scale = 1.0)
param_v = mle(grid, y, kernel_v, theta_init, bnds)

#%%

print(param_u, param_v)


#%%


    

#%% Choleksy factors

kernel_u = Matern(nu = 0.5, length_scale = 0.1) 
kernel_v = Matern(nu = 0.5, length_scale = 0.1)




#%% Cholesky transformation




#%% Optimal recovery: pointwise values

u_recov = []
for i in range(x.shape[0]):
    u_recov.append(K@L_K_inv@x_train[i])
u_recov = np.array(u_recov)

    
v_recov = []
for i in range(y.shape[0]):
    v_recov.append(G@L_G_inv@pred_train[i])
v_recov = np.array(v_recov)

e_u = np.mean(np.linalg.norm(u_recov - x, axis = -1)/np.linalg.norm(x, axis = -1))
e_v = np.mean(np.linalg.norm(v_recov - y, axis = -1)/np.linalg.norm(y, axis = -1))

print(e_u, e_v)

u_recov = []
for i in range(x_test.shape[0]):
    u_recov.append(K@L_K_inv@x_val[i])
u_recov = np.array(u_recov)

    
v_recov = []
for i in range(y_test.shape[0]):
    v_recov.append(G@L_G_inv@y_val[i])
v_recov = np.array(v_recov)

e_u = np.mean(np.linalg.norm(u_recov - x_test, axis = -1)/np.linalg.norm(x_test, axis = -1))
e_v = np.mean(np.linalg.norm(v_recov - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(e_u, e_v)


#%%

pred, gp = train_test(x, x_test, y_train, y_val)
pred_train = gp.predict(x)
e = np.mean(np.linalg.norm(pred - y_val, axis = -1)/np.linalg.norm(y_val, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y_train, axis = -1)/np.linalg.norm(y_train, axis = -1))

print(e, e_train)

print(gp.kernel_)

#%% Recovering the pointwise measurements


pred_point = []
for i in range(y_test.shape[0]):
    pred_point.append(G@L_G_inv@pred[i])
    
pred_point = np.array(pred_point)
    
pred_point_train = []
for i in range(x.shape[0]):
    pred_point_train.append(G@L_G_inv@pred_train[i])
pred_point_train = np.array(pred_point_train)

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print(e, e_train)

idx = 15

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))
ax1.plot(grid, y[idx], label = "True")
ax1.plot(grid, pred_point_train[idx], label = "Prediction")
ax1.set_xlabel(r'$x$', size= 15)
ax1.set_ylabel(r'$u(x,1)$', size= 15)
ax1.set_title("Prediction on the training set", size = 15)
ax1.legend()

ax2.plot(grid, y_test[idx],  label = "True")
ax2.plot(grid, pred_point[idx],  label = "Prediction")
ax2.set_xlabel(r'$x$', size= 15)
ax2.set_ylabel(r'$u(x, 1)$', size= 15)
ax2.set_title("Prediction on the test set", size = 15)
ax2.legend()
#%%



# #%%
# def log_marginal_likelihood(y, kernel, theta):
#     kernel.length_scale =theta
#     #print(kernel)
#     K, grad_K = kernel(grid, eval_gradient = True)
    
#     #print(K.shape)
#     #  Compute the log likelihood
#     L = cholesky(K + 1e-10*np.eye(K.shape[0]), lower = True, check_finite=True)
#     #print(L)
#     alpha = scipy.linalg.solve(K + 1e-10*np.eye(K.shape[0]), y.T,  assume_a='pos').T
#     print(alpha)

#     #print((y*alpha).shape)
#     log_p = -0.5*np.sum(y*alpha, axis = -1)
#     log_p -= 0.5*np.log(np.diag(L)).sum()
#     log_p -=K.shape[0] / 2 * np.log(2 * np.pi)
    
#     # Compute the gradient 
#     grad = 0
#     # #print(grad_K.shape, y.shape)
#     # temp = (np.squeeze(grad_K) @ alpha[:, :, None]).squeeze(-1)
#     # #print(temp.shape)
#     # #print(K.shape, temp.shape)
#     # alpha_2 = scipy.linalg.solve(K+ 1e-10*np.eye(K.shape[0]), temp.T, assume_a = 'pos').T
#     # #print(alpha_2.shape)
    
#     # grad = 0.5*np.sum(np.sum(y*alpha_2, axis = -1))
#     # grad = -0.5*y.shape[0]*np.trace(scipy.linalg.solve(K+ 1e-10*np.eye(K.shape[0]), grad_K, assume_a= "pos"))
    
#     return log_p, grad
    

# print(log_marginal_likelihood(x[:1], kernel_u, theta_test))

# #%%

# def obj_func(theta, kernel, y):
    
#     value, grad = log_marginal_likelihood(y, kernel, theta)
#     return -value, -grad


# #%%

# # initial_theta = np.array([10.0])
# # bnds =  ((0.0, 10))
# # opt_res = scipy.optimize.minimize(
# #                 obj_func,
# #                 initial_theta,
# #                 args = (kernel_u, x),
# #                 method="BFGS",
# #                 jac=True)

# #%%

# # print(opt_res.x)