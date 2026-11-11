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
X_true = data['X_matrix'].flatten()[:, None]
rho0_true = data['rho0_matrix'].flatten()[:, None]
v_true = data['v_matrix'].flatten()[:, None]

#Nt = t.shape[0] # = r.shape[0]
Nr = r.shape[1]

# t ∈ [lb[0], ub[0]], r ∈ [lb[1], ub[1]]
lb = np.array([0.0, 0.0])
ub = np.array([t[-1].item(), r[0:1, -1].item()])

model = NN_arch(layers, lb, ub)

lhs_sampler = Lhs(criterion="maximin", iterations=1000)
unit = Real(0.0, 1.0)
space = [unit, unit]

# --- Initial points sampling ---

N0 = 50 # Number of initial points
lhs_unit_ic = lhs_sampler.generate(dimensions=[unit], n_samples=N0)
# Convert the list of lists to a NumPy array
lhs_unit_ic = np.array(lhs_unit_ic)
r0 = lb[1] + (ub[1] - lb[1]) * lhs_unit_ic

# --- Boundary points sampling ---

# Number of boundary points = todos los instantes de tiempo considerados (Nb=51)
tb = t
Rtb = r[:,-1]

# --- Collocation points sampling ---

Nc = 5000 # Aumentarlo cuando ya el código sea el definitivo
lhs_unit = lhs_sampler.generate(dimensions=space, n_samples=Nc) # lhs_unit es un array 2D N_f x 2 con puntos en [0,1]^2, distribuidos con Latin Hypercube Sampling.
# Esto asegura que cada variable cubra su dominio uniformemente, sin repetición aleatoria, lo que mejora la exploración del espacio comparado con np.random.rand
# Convert the list of lists to a NumPy array
lhs_unit = np.array(lhs_unit)
Xc = lb + (ub - lb) * lhs_unit # X_f tiene dimensión (N_f, 2) con puntos uniformemente dispersos en el dominio 2D t ∈ [lb[0], ub[0]], r ∈ [lb[1], ub[1]]



### A POR TODAS ###

Xstar = np.hstack((
np.repeat(t, Nr, axis=0),  # → shape (Nt*Nr, 1). Repite cada t[i] Nr veces
r.reshape(-1, 1)   # → shape (Nt*Nr, 1). Asegura que r sea columna
)) # resultado → shape (Nt*Nr, 2)
# np.hstack() espera una secuencia (lista o tupla) de arrays como argumento: np.hstack([a, b]) o np.hstack((a, b)). 
# Los paréntesis internos (...) crean una tupla: (array1, array2)

pinn = PhysicsInformedModel(model, r0, tb, Rtb, Xc, M)
pinn.train(epochs=1000) # Aumentar el número de epochs cuando ya el código sea el definitivo

U_pred = pinn.predict(Xstar)
error_alpha = np.linalg.norm(alpha_true-U_pred[:, 0:1],2)/np.linalg.norm(alpha_true,2)
error_X = np.linalg.norm(X_true-U_pred[:, 1:2],2)/np.linalg.norm(X_true,2)
error_rho0 = np.linalg.norm(rho0_true-U_pred[:, 2:3],2)/np.linalg.norm(rho0_true,2)
error_v = np.linalg.norm(v_true-U_pred[:, 3:4],2)/np.linalg.norm(v_true,2)
print('Relative error alpha: %e' % (error_alpha))
print('Relative error X: %e' % (error_X))
print('Relative error rho0: %e' % (error_rho0))
print('Relative error v: %e' % (error_v))

np.savez('./results/OS_pred_def_de_verdad_de_la_buena.npz', U_pred=U_pred)