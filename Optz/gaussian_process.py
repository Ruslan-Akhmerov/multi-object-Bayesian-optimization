import numpy as np
from scipy.linalg import cholesky, solve_triangular

def covariance_matrix(X1, X2, length_scale=1.0):
    return np.exp(-0.5 * np.subtract.outer(X1, X2)**2 / length_scale**2)

def update_gaussian_process(X_train, y_train, X_test):
    length_scale=1.0
    noise_level = 0.01
    K = covariance_matrix(X_train, X_train, length_scale)
    K_noise = K + noise_level**2 * np.eye(len(X_train))
    L = np.linalg.cholesky(K_noise)
    
    K_pred = covariance_matrix(X_train, X_test, length_scale)
    Lk = np.linalg.solve(L, K_pred)
    
    mu = np.dot(Lk.T, np.linalg.solve(L, y_train))
    cov = covariance_matrix(X_test, X_test, length_scale) - np.dot(Lk.T, Lk)
    
    return mu, cov

