# -*- coding: utf-8 -*-
import numpy as np
import scipy.io

M = 1

data = scipy.io.loadmat('./main/Data/OS.mat')

r = data['r_matrix']
alpha_star = data['alpha_matrix']
X_star = data['X_matrix']
rho0_star = data['rho0_matrix']
v_star = data['v_matrix']

Nt = r.shape[0] # = r.shape[0]
Nr = r.shape[1]
# print([Nt,Nr])


# -------------------------------

# alpha_sup = alpha_star[:,-1]
# print(alpha_sup)
# alpha_potencial = np.exp(- M / r[:,-1])
# print(alpha_potencial)

# print(alpha_sup-alpha_potencial)
# # --> NO SALE LO MISMO

# -------------------------------

# alpha_0 = alpha_star[0,:]
# print(alpha_0)
# print(r[0,:])
# alpha_potencial_0 = np.exp(- M * (3 * (10*M)**2 - r[0,:]) / (2 * (10*M)**3))
# print(alpha_potencial_0)

# print(alpha_0-alpha_potencial_0)
# # --> NO SALE LO MISMO

# -------------------------------

# print(v_star[0,:])
# print(0.5 * r[0,:] * (1 - 1 / X_star[0,:]**2) - M*(r[0,:]/r[0,-1])**3)
# print(rho0_star[0,:] - 3*M/(4*np.pi*r[0,-1]**3))
# # --> SÍ SALE LO MISMO

# -------------------------------

# print(X_star)

# print(np.var(alpha_star))
# print(np.var(X_star))
# print(np.var(rho0_star))
# print(np.var(v_star))

# print(data['m_matrix'])

# print(r[0:1,:]/r[0,-1])

# print(r[:,-1])

# -------------------------------

Nc = 5000 # Aumentarlo cuando ya el código sea el definitivo

R = r[:,-1]

i_vals = np.arange(1, Nt)       # i in 2..Nt (índice Python 1..Nt-1)
j_vals = np.arange(1, Nr - 1)   # j in 2..Nr-1 (índice Python 1..Nr-2)

# Todos los puntos válidos
points = np.array([(i, j) for i in i_vals for j in j_vals])

# # Calcular pesos de sesgo hacia abajo a la izquierda
# weights = np.array([i * (Nr - j) for i, j in points], dtype=np.float64)
# weights /= weights.sum()  # normalizar

# Selección aleatoria ponderada
indices = np.random.choice(len(points), size=Nc, replace=False) #, p=weights)
selected_points = points[indices]

import matplotlib.pyplot as plt

# Visualización
fig, ax = plt.subplots(figsize=(12, 6))  # Tamaño adecuado para ejes grandes
im = ax.imshow(r, cmap='viridis', origin='upper', aspect='auto')

# Dibujar puntos seleccionados
filas, columnas = np.array(selected_points[:, 0]), np.array(selected_points[:, 1])
ax.scatter(columnas, filas, color='red', s=10, edgecolors='white', label='Seleccionados')

# Opciones para evitar solapamientos
ax.set_xticks(np.linspace(0, Nr-1, 10))  # Solo 10 ticks en X
ax.set_yticks(np.linspace(0, Nt-1, 10))  # Solo 10 ticks en Y
ax.set_xlabel("Columnas (j)")
ax.set_ylabel("Filas (i)")
ax.set_title("Puntos de colocación seleccionados sobre la malla de validación")

# Mostrar leyenda y barra de color
ax.legend(loc='upper right')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'$r_{i,j}$')

# Optimizar espaciado
plt.tight_layout()
plt.show()

print(r[0,:])

# # print(filas)

# -------------------------------