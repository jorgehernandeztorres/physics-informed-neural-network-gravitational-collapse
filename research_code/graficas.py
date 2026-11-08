# -*- coding: utf-8 -*-
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Carga de datos
mat_data = scipy.io.loadmat('./main/Data/OS.mat')
pred_data = np.load('./results/OS_pred_def_de_verdad_3.npz')

# Variables del .mat
m_mat = mat_data['m_matrix']
v_mat = mat_data['v_matrix']
rho0_mat = mat_data['rho0_matrix']
alpha_mat = mat_data['alpha_matrix']
X_mat = mat_data['X_matrix']
t_vec = mat_data['t_vector'].flatten()
r_mat = mat_data['r_matrix']

Nt, Nr = alpha_mat.shape
M = 1.0
r_2M = r_mat / (2 * M)

# Predicciones de la PINN
U_pred = pred_data['U_pred']  # shape: (Nt*Nr, 4)
alpha_pred = U_pred[:, 0].reshape(Nt, Nr)
X_pred     = U_pred[:, 1].reshape(Nt, Nr)
rho0_pred  = U_pred[:, 2].reshape(Nt, Nr)
v_pred     = U_pred[:, 3].reshape(Nt, Nr)

# Calcular m_pred a partir de X_pred y r
m_pred = 0.5 * r_mat * (1 - 1 / X_pred**2)
# eps = 1e-8
# m_pred = 0.5 * r_mat * (1 - 1 / np.clip(X_pred**2, eps, None))

# Tiempos que se usarán para graficar (0, Nt//8, ..., Nt-1)
idx_times = np.linspace(0, Nt - 1, 9, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, 9))  # paleta de colores para distinguir curvas

def plot_variable(var_matrix, pred_matrix, var_name):
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(idx_times):
        r_vals = r_2M[idx, :]
        theor_vals = var_matrix[idx, :]
        pred_vals = pred_matrix[idx, :]

        # Curva teórica (suavizada)
        interp = interp1d(r_vals, theor_vals, kind='cubic', fill_value="extrapolate")
        r_dense = np.linspace(r_vals.min(), r_vals.max(), 500)
        theor_smooth = interp(r_dense)
        plt.plot(r_dense, theor_smooth, color=colors[i], label=f'Teórica t={t_vec[idx]:.2f}')
        
        # Predicción de la PINN (discreta)
        plt.scatter(r_vals, pred_vals, color=colors[i], marker='x', label=f'PINN t={t_vec[idx]:.2f}')

    plt.xlabel(r'$r/(2M)$')
    plt.ylabel(var_name)
    plt.title(f'Comparación entre solución teórica y PINN para {var_name}')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Graficar cada variable
plot_variable(X_mat, X_pred, r'$X$')
plot_variable(alpha_mat, alpha_pred, r'$\alpha$')
plot_variable(m_mat, m_pred, r'$m$')
plot_variable(rho0_mat, rho0_pred, r'$\rho_0$')
plot_variable(v_mat, v_pred, r'$v$')