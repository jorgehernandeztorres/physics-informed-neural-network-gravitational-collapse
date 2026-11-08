import numpy as np
import scipy.io

# Carga de datos
mat_data = scipy.io.loadmat('./main/Data/OS.mat')
pred_data = np.load('./results/OS_pred_def_de_verdad_de_la_buena.npz')

# Variables del .mat
m_mat = mat_data['m_matrix']
v_mat = mat_data['v_matrix']
rho0_mat = mat_data['rho0_matrix']
alpha_mat = mat_data['alpha_matrix']
X_mat = mat_data['X_matrix']
t_vec = mat_data['t_vector'].flatten()
r_mat = mat_data['r_matrix']

Nt, Nr = alpha_mat.shape

# Predicciones de la PINN
U_pred = pred_data['U_pred']  # shape: (Nt*Nr, 4)
alpha_pred = U_pred[:, 0].reshape(Nt, Nr)
X_pred     = U_pred[:, 1].reshape(Nt, Nr)
rho0_pred  = U_pred[:, 2].reshape(Nt, Nr)
v_pred     = U_pred[:, 3].reshape(Nt, Nr)

# Calcular m_pred a partir de X_pred y r
m_pred = 0.5 * r_mat * (1 - 1 / X_pred**2)

# print(m_pred)

error_rho0 = np.linalg.norm(rho0_mat-rho0_pred,2)/np.linalg.norm(rho0_mat,2)
print(error_rho0)