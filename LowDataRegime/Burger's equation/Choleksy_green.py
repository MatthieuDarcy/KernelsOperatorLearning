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
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel




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
    gp = GaussianProcessRegressor(kernel, alpha = 1e-10,  normalize_y = False, random_state= 6032023) 
    
    gp.fit(x_train, y_train)
    pred= gp.predict(x_test)
    #pred_train = gp.predict(x_train)

    #e = compute_error_dataset(y_test, pred, knots, k)

    return pred, gp

pred, GP = train_test(x, x_test, y, y_test)
pred_train = GP.predict(x)
e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print("Error without cholesky preconditioning")
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

ax2.plot(grid, y_test[idx],  label = "True")
ax2.plot(grid, pred[idx],  label = "Prediction")
ax2.set_xlabel(r'$x$', size= 15)
ax2.set_ylabel(r'$u(x, 1)$', size= 15)
ax2.set_title("Prediction on the test set", size = 15)
ax2.legend()



#%% Choelsky precondition

kernel_u =Matern(nu = 1.5, length_scale = 0.1) 
kernel_v = Matern(nu = 1.5, length_scale = 0.1)
def cholesky_transform(kernel_u, kernel_v, grid, u, v):
    K = kernel_u(grid)
    G = kernel_v(grid)

    tau = 1e-8
    L_K = cholesky(K + tau*np.eye(K.shape[0]), lower=True)
    L_G = cholesky(G+ tau*np.eye(K.shape[0]), lower = False)

    tau = 1e-8
    #L_K_inv = np.linalg.inv(L_K + tau*np.eye(K.shape[0]))
    L_G_inv = np.linalg.inv(L_G + tau*np.eye(K.shape[0]))

    
    return (L_K.T @ u[:, :, None]).squeeze(-1), (L_G_inv.T @ v[:, :, None]).squeeze(-1)

x_train, y_train = cholesky_transform(kernel_u, kernel_v, grid, x, y)
x_val, y_val = cholesky_transform(kernel_u, kernel_v, grid, x_test, y_test)


#%% Optimal recovery: pointwize values

def optimal_recovery(kernel_u, kernel_v, grid, u, v):
    K = kernel_u(grid)
    G = kernel_v(grid)
    #print(np.linalg.cond(K))

    tau = 1e-8
    tau = 0
    L_K = cholesky(K + tau*np.eye(K.shape[0]), lower=True)
    L_G = cholesky(G+ tau*np.eye(K.shape[0]), lower = False)

    tau = 1e-8
    L_K_inv = np.linalg.inv(L_K + tau*np.eye(K.shape[0]))
    L_G_inv = np.linalg.inv(L_G + tau*np.eye(K.shape[0]))
    
    u = np.linalg.solve(L_K.T, u[:, :, None]).squeeze(-1)
    #u_recov = np.squeeze(K@scipy.linalg.cho_solve((L_K, True), u[None] ))
    u_recov = np.squeeze(K@scipy.linalg.solve(K, u.T, assume_a = 'pos' )).T
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


#%%

pred, gp = train_test(x_train, x_val, y_train, y_val)
pred_train = gp.predict(x_train)
e = np.mean(np.linalg.norm(pred - y_val, axis = -1)/np.linalg.norm(y_val, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y_train, axis = -1)/np.linalg.norm(y_train, axis = -1))


print(e, e_train)

print(gp.kernel_)

#%% Recovering the pointwise measurements

_, pred_point_train = optimal_recovery(kernel_u, kernel_v, grid, x_train, pred_train)
_, pred_point = optimal_recovery(kernel_u, kernel_v, grid, x_val, pred)

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print("Error with cholesky preconditioning")
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

#%% Only preconditioning the output

pred, gp = train_test(x, x_test, y_train, y_val)
pred_train = gp.predict(x)
e = np.mean(np.linalg.norm(pred - y_val, axis = -1)/np.linalg.norm(y_val, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y_train, axis = -1)/np.linalg.norm(y_train, axis = -1))


print(e, e_train)

print(gp.kernel_)

#%% Recovering the pointwise measurements

_, pred_point_train = optimal_recovery(kernel_u, kernel_v, grid, x_train, pred_train)
_, pred_point = optimal_recovery(kernel_u, kernel_v, grid, x_val, pred)

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print("Error with cholesky preconditioning on the output")
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
                    options = {"disp": True})
    print(opt_res.message)
    return opt_res.x

#%%
# theta_init = np.array([1.0, 1.0])
# bnds =  ((1e-5, 1e5),(1e-5, 1e5))
# kernel_u = ConstantKernel()*Matern(nu = 0.5, length_scale = 1.0)
# param_u = mle(grid, x, kernel_u, theta_init, bnds)

# #%%
# theta_init = np.array([1.0, 1.0])
# bnds =  ((1e-5, 1e5),(1e-5, 1e5))
# kernel_v = ConstantKernel()*Matern(nu = 0.5, length_scale = 1.0)
# param_v = mle(grid, y, kernel_v, theta_init, bnds)


#%%
theta_init = np.array([1.0])
bnds =  ((1e-5, 1e5),)
kernel_u = Matern(nu = 1.5, length_scale = 1.0)
param_u = mle(grid, x, kernel_u, theta_init, bnds)

#%%
theta_init = np.array([1.0])
bnds =  ((1e-5, 1e5),)
kernel_v = Matern(nu = 0.5, length_scale = 1.0)
param_v = mle(grid, y, kernel_v, theta_init, bnds)

#%%

print(param_u, param_v)


#%%
kernel_u = Matern(nu = 1.5, length_scale = param_u[-1]) 
kernel_v = Matern(nu = 0.5, length_scale = param_v[-1])

x_train, y_train = cholesky_transform(kernel_u, kernel_v, grid, x, y)
x_val, y_val = cholesky_transform(kernel_u, kernel_v, grid, x_test, y_test)


#%% Validating the recovery

u_recov, v_recov = optimal_recovery(kernel_u, kernel_v, grid, x_train, y_train)
    
e_u = np.mean(np.linalg.norm(u_recov - x, axis = -1)/np.linalg.norm(x, axis = -1))
e_v = np.mean(np.linalg.norm(v_recov - y, axis = -1)/np.linalg.norm(y, axis = -1))

print(e_u, e_v)


u_recov, v_recov = optimal_recovery(kernel_u, kernel_v, grid, x_val, y_val)
    
e_u = np.mean(np.linalg.norm(u_recov - x_test, axis = -1)/np.linalg.norm(x_test, axis = -1))
e_v = np.mean(np.linalg.norm(v_recov - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(e_u, e_v)



#%%
pred, gp = train_test(x_train, x_val, y_train, y_val)
pred_train = gp.predict(x_train)

e = np.mean(np.linalg.norm(pred - y_val, axis = -1)/np.linalg.norm(y_val, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y_train, axis = -1)/np.linalg.norm(y_train, axis = -1))

print(e, e_train)
print(gp.kernel_)



#%%

_, pred_point_train = optimal_recovery(kernel_u, kernel_v, grid, x_train, pred_train)
_, pred_point = optimal_recovery(kernel_u, kernel_v, grid, x_val, pred)

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print("Error with cholesky preconditioning + MLE")
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

#%% Only preconditioning the output

pred, gp = train_test(x, x_test, y_train, y_val)
pred_train = gp.predict(x)
e = np.mean(np.linalg.norm(pred - y_val, axis = -1)/np.linalg.norm(y_val, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y_train, axis = -1)/np.linalg.norm(y_train, axis = -1))


print(e, e_train)

print(gp.kernel_)

#%% Recovering the pointwise measurements

_, pred_point_train = optimal_recovery(kernel_u, kernel_v, grid, x_train, pred_train)
_, pred_point = optimal_recovery(kernel_u, kernel_v, grid, x_val, pred)

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print("Error with cholesky preconditioning on the output + MLE")
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

#%% Using the coefficients in the basis


# Here we predict the coefficients in the basis instead of the pointwise values
K_v = kernel_v(grid)

c_train = scipy.linalg.solve(K_v + 1e-10*np.eye(K_v.shape[0]), y.T).T
c_test = scipy.linalg.solve(K_v + 1e-10*np.eye(K_v.shape[0]), y_test.T).T

#%%

def train_test(x_train, x_test, y_train, y_test):
    kernel = Matern(nu = 2.5, length_scale = 1.0)
    gp = GaussianProcessRegressor(kernel, alpha = 1e-10,  normalize_y = False, random_state= 6032023) 
    
    gp.fit(x_train, y_train)
    pred= gp.predict(x_test)
    #pred_train = gp.predict(x_train)

    #e = compute_error_dataset(y_test, pred, knots, k)

    return pred, gp


pred, gp = train_test(x, x_test, c_train, c_test)
pred_train = gp.predict(x)

e = np.mean(np.linalg.norm(pred - c_test, axis = -1)/np.linalg.norm(c_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - c_train, axis = -1)/np.linalg.norm(c_train, axis = -1))

print(e, e_train)
print(gp.kernel_)


#%%
pred_point = (K_v@pred[:, :, None]).squeeze()
pred_point_train = (K_v@pred_train[:, :, None]).squeeze()

# pred_point = (K_v@c_test[:, :, None]).squeeze()
# pred_point_train = (K_v@c_train[:, :, None]).squeeze()

#%%

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print("Error by predicting the coefficeint of the output + MLE")
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


#%% Using the coefficients in the basis for the input + output
kernel_u =Matern(nu = 0.5, length_scale = param_u[-1]) 
kernel_v = Matern(nu = 0.5, length_scale = param_v[-1])

x_train, y_train = cholesky_transform(kernel_u, kernel_v, grid, x, y)
x_val, y_val = cholesky_transform(kernel_u, kernel_v, grid, x_test, y_test)

#%%

# Here we predict the coefficients in the basis instead of the pointwise values
K_v = kernel_v(grid)

c_train = scipy.linalg.solve(K_v + 1e-10*np.eye(K_v.shape[0]), y.T).T
c_test = scipy.linalg.solve(K_v + 1e-10*np.eye(K_v.shape[0]), y_test.T).T

#%%

pred, gp = train_test(x_train, x_val, c_train, c_test)
pred_train = gp.predict(x_train)

e = np.mean(np.linalg.norm(pred - c_test, axis = -1)/np.linalg.norm(c_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - c_train, axis = -1)/np.linalg.norm(c_train, axis = -1))

print(e, e_train)
print(gp.kernel_)

#%% Going back to the pointwise values  
pred_point = (K_v@pred[:, :, None]).squeeze()
pred_point_train = (K_v@pred_train[:, :, None]).squeeze()

#pred_point = (K_v@c_test[:, :, None]).squeeze()
#pred_point_train = (K_v@c_train[:, :, None]).squeeze()

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print("Error by predicting the coefficeint + MLE")
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


#%% Using the coefficients in the basis for the input + output
kernel_u =Matern(nu = 0.5, length_scale = param_u[-1]) 
kernel_v = Matern(nu = 0.5, length_scale = param_v[-1])

x_train, y_train = cholesky_transform(kernel_u, kernel_v, grid, x, y)
x_val, y_val = cholesky_transform(kernel_u, kernel_v, grid, x_test, y_test)


pred, gp = train_test(x_train, x_val, y, y_test)
pred_train = gp.predict(x_train)

e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print(e, e_train)
print(gp.kernel_)

#%% Going back to the pointwise values  
pred_point = pred
pred_point_train = pred_train

#pred_point = (K_v@c_test[:, :, None]).squeeze()
#pred_point_train = (K_v@c_train[:, :, None]).squeeze()

e = np.mean(np.linalg.norm(pred_point - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
e_train = np.mean(np.linalg.norm(pred_point_train - y, axis = -1)/np.linalg.norm(y, axis = -1))

print("Error by predicting the coefficeint + MLE")
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




# #%% Pairwise diff: checking our cholesky factors


# def rkhs_norm(x,y, L_K):
    
#     diff = (x -y)
#     return diff@scipy.linalg.cho_solve(L_K, diff)

# K_u = kernel_u(grid)
# L_K = scipy.linalg.cho_factor(K_u)

# #print(rkhs_norm(x[0], x[1], L_K))

# callable_f = lambda x,y : rkhs_norm(x,y, L_K)

# from sklearn.metrics import pairwise_distances
# dist = pairwise_distances(x,x, metric = callable_f)

# #%%
# kernel_test = Matern(nu = 0.5, length_scale = 1.0)

# x_train, y_train = cholesky_transform(kernel_u, kernel_v, grid, x, y)
# x_val, y_val = cholesky_transform(kernel_u, kernel_v, grid, x_test, y_test)

# K_test = kernel_test(x_train) 

# #%%


# K_test_2 = np.exp(-np.sqrt(dist))


# print(np.mean((K_test_2 - K_test)**2))

# #%%



# #%%

# def cholesky_transform(kernel_u, kernel_v, grid, u, v):
#     K = kernel_u(grid)
#     G = kernel_v(grid)
    
#     tau = 1e-10
#     L_K = cholesky(K+ tau*np.eye(K.shape[0]), lower=False)
#     L_G = cholesky(G + tau*np.eye(K.shape[0]), lower = False)

#     tau = 1e-10
#     L_K_inv = np.linalg.inv(L_K + tau*np.eye(K.shape[0]))
#     L_G_inv = np.linalg.inv(L_G + tau*np.eye(K.shape[0]))
    
#     return (L_K_inv.T @ u[:, :, None]).squeeze(-1), (L_G_inv.T @ v[:, :, None]).squeeze(-1)
    
# x_train, y_train = cholesky_transform(kernel_u, kernel_v, grid, x, y)
    
# #%%

# def rkhs_norm_point(x, L_K):
    

#     return x@scipy.linalg.cho_solve(L_K, x)

# x_train, y_train = cholesky_transform(kernel_u, kernel_v, grid, x, y)
# x_val, y_val = cholesky_transform(kernel_u, kernel_v, grid, x_test, y_test)

# a = x_train[0].T@x_train[0]
# b = rkhs_norm_point(x[0], L_K)

# K_u = kernel_u(grid)
# L_K = scipy.linalg.cho_factor(K_u)
# print(a,b)

# #%%

# L_K_inv = np.linalg.inv(scipy.linalg.cholesky(K_u+ 1e-8*np.eye(K_u.shape[0])))
# #%%


# #print(scipy.linalg.cho_solve(L_K, x[0]), L_K_inv@L_K_inv.T@x[0])

# print(x[0].T@scipy.linalg.cho_solve(L_K, x[0]), (L_K_inv.T@x[0])@L_K_inv.T@x[0])

# print(x_train[0]@x_train[0])









