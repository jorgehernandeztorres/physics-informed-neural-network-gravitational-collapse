# -*- coding: utf-8 -*-
import numpy as np
import scipy.io
from scipy.optimize import minimize
from skopt.sampler import Lhs # skopt.sampler es una librería que proporciona funciones de muestreo experimental, como Latin Hypercube Sampling (LHS)
from skopt.space import Real # Real(low, high) define una dimensión continua en el espacio de búsqueda

from PINN_OS import PhysicsInformedModel, NN_arch

# PARÁMETROS DE OS.mat: Nt = 51, Nr = 500, M = 1, R0 = 10, t_max = 50

M = 1

hidden_layers = 4
neurons = 100 # Aumentarlo cuando ya el código sea el definitivo
layers = [2] + [neurons]*hidden_layers + [4] # Esto significa que la red espera como entrada un tensor de shape (N, 2): cada fila representa un punto con dos variables independientes
# — en este caso, tiempo t y radio r.
# 👉 En PyTorch, cada muestra de entrada es una fila del tensor, y cada columna representa una característica (feature). En tu caso: x = [t, r].
# También la red neuronal espera salidas en formato 2D (batch, dimensión), con lo que hay que tenerlo en cuenta.

data = scipy.io.loadmat('./main/Data/OS.mat')

t = data['t_vector'].flatten()[:, None] # de tamaño (Nt,1)
r = data['r_matrix'] # matriz de tamaño (Nt,Nr)
alpha_true = data['alpha_matrix'].flatten()[:, None] # de tamaño (Nt*Nr,1)
m_true = data['m_matrix'].flatten()[:, None]
rho0_true = data['rho0_matrix'].flatten()[:, None]
v_true = data['v_matrix'].flatten()[:, None]

Nt = t.shape[0] # = r.shape[0]
Nr = r.shape[1]

# t ∈ [lb[0], ub[0]], r ∈ [lb[1], ub[1]]
lb = np.array([0.0, 0.0])
ub = np.array([t[-1].item(), r[0:1, -1].item()])

R0 = r[0,-1]

model = NN_arch(layers, lb, ub, M, R0)

lhs_sampler = Lhs(criterion="maximin", iterations=1000)
unit = Real(0.0, 1.0)
space = [unit, unit]

# --- Initial points sampling ---

N0 = 50 # Number of initial points
log_r0 = np.logspace(np.log10(1e-3), np.log10(ub[1]), num=N0)
r0 = log_r0[:, None]  # shape (N0, 1)

# --- Boundary points sampling ---

# Number of boundary points = todos los instantes de tiempo considerados (Nb=51)
tb = t
Rtb = r[:,-1]

# --- Collocation points sampling ---

Nc = 8000 # Aumentarlo cuando ya el código sea el definitivo

R = r[:,-1]

i_vals = np.arange(1, Nt)       # i in 2..Nt (índice Python 1..Nt-1)
j_vals = np.arange(1, Nr - 1)   # j in 2..Nr-1 (índice Python 1..Nr-2)

# Todos los puntos válidos
points = np.array([(i, j) for i in i_vals for j in j_vals])

# Calcular pesos de sesgo hacia abajo a la izquierda
weights = np.array([i * (Nr - j) for i, j in points], dtype=np.float64)
weights /= weights.sum()  # normalizar

# Selección aleatoria ponderada
indices = np.random.choice(len(points), size=Nc, replace=False, p=weights)
selected_points = points[indices]

# Obtener filas y columnas por separado
filas, columnas = selected_points[:, 0], selected_points[:, 1]

# Extraer los valores de la matriz
valores = r[filas, columnas]

# Construir matriz Nc x 2
Xc = np.column_stack((t[filas], valores))

# lhs_unit = lhs_sampler.generate(dimensions=space, n_samples=Nc)
# lhs_unit = np.array(lhs_unit)

# # Reescalar r con densidad logarítmica hacia el centro
# lhs_unit[:,1] = lhs_unit[:,1]**1.5  # Elevar el componente radial para sesgar hacia r pequeño
# Xc = lb + (ub - lb) * lhs_unit



### A POR TODAS ###

Xstar = np.hstack((
np.repeat(t, Nr, axis=0),  # → shape (Nt*Nr, 1). Repite cada t[i] Nr veces
r.reshape(-1, 1)   # → shape (Nt*Nr, 1). Asegura que r sea columna
)) # resultado → shape (Nt*Nr, 2)
# np.hstack() espera una secuencia (lista o tupla) de arrays como argumento: np.hstack([a, b]) o np.hstack((a, b)). 
# Los paréntesis internos (...) crean una tupla: (array1, array2)

pinn = PhysicsInformedModel(model, r0, tb, Rtb, Xc, M)
pinn.train(epochs=5000) # Aumentar el número de epochs cuando ya el código sea el definitivo

U_pred = pinn.predict(Xstar)
error_alpha = np.linalg.norm(alpha_true-U_pred[:, 0:1],2)/np.linalg.norm(alpha_true,2)
error_m = np.linalg.norm(m_true-U_pred[:, 1:2],2)/np.linalg.norm(m_true,2)
error_rho0 = np.linalg.norm(rho0_true-U_pred[:, 2:3],2)/np.linalg.norm(rho0_true,2)
error_v = np.linalg.norm(v_true-U_pred[:, 3:4],2)/np.linalg.norm(v_true,2)
print('Relative error alpha: %e' % (error_alpha))
print('Relative error m: %e' % (error_m))
print('Relative error rho0: %e' % (error_rho0))
print('Relative error v: %e' % (error_v))

np.savez('./results/OS_pred_def_de_verdad_3.npz', U_pred=U_pred)