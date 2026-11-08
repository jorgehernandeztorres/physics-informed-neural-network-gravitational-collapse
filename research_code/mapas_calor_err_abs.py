import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Cargar predicciones
data = np.load("./results/OS_pred_def_de_verdad_de_la_buena.npz")
U_pred = data["U_pred"]  # shape: (Nt*Nr, 4)

# Cargar soluciones analíticas
mat = scipy.io.loadmat("./main/Data/OS.mat")
t = mat['t_vector'].flatten()[:, None] # de tamaño (Nt,1)
r = mat['r_matrix'] # matriz de tamaño (Nt,Nr)
alpha_true = mat['alpha_matrix']
m_true = mat['m_matrix']
rho0_true = mat['rho0_matrix']
v_true = mat['v_matrix']

Nt, Nr = alpha_true.shape

# Cargar predicciones
alpha_pred = U_pred[:, 0].reshape(Nt, Nr)
X_pred     = U_pred[:, 1].reshape(Nt, Nr)
rho0_pred  = U_pred[:, 2].reshape(Nt, Nr)
v_pred     = U_pred[:, 3].reshape(Nt, Nr)

# Calcular m
m_pred = 0.5 * r * (1 - 1 / X_pred**2)

# Error absoluto puntual
def absolute_error(pred, true):
    eps = 1e-10
    return np.abs(pred - true)

err_alpha = absolute_error(alpha_pred, alpha_true)
err_m = absolute_error(m_pred, m_true)
err_rho0 = absolute_error(rho0_pred, rho0_true)
err_v = absolute_error(v_pred, v_true)

# Crear mallas (no uniformes en r) para cada instante de tiempo
T_plot = np.zeros_like(r)
R_plot = r.copy()  # ya contiene r desde 0 hasta R(t) para cada fila

for i in range(Nt):
    T_plot[i, :] = t[i]  # Repetir t_i a lo largo de la fila

# Expandimos r y t para construir mallas de tamaño (Nt+1, Nr+1)
# Necesario para que pcolormesh asigne correctamente los bordes de las celdas
def expand_edges(mat):
    """Construye una matriz (N+1) interpolando bordes entre puntos para pcolormesh"""
    extended = np.zeros((mat.shape[0]+1, mat.shape[1]+1))
    extended[:-1, :-1] = mat
    extended[-1, :-1] = mat[-1, :]
    extended[:-1, -1] = mat[:, -1]
    extended[-1, -1] = mat[-1, -1]
    return extended

R_plot_ext = expand_edges(R_plot)
T_plot_ext = expand_edges(T_plot)

# Ahora sí: gráfica precisa
def plot_heatmap_variable_r_preciso(err_ext, R_ext, T_ext, title_latex):
    plt.figure(figsize=(8, 6))
    pcm = plt.pcolormesh(R_ext, T_ext, err_ext, shading='auto', cmap='viridis')
    plt.colorbar(pcm, label='Error absoluto')
    plt.xlabel('r')
    plt.ylabel('t')
    plt.title(f"Error absoluto de ${title_latex}$", fontsize=14)
    plt.tight_layout()
    plt.show()

# Uso
plot_heatmap_variable_r_preciso(err_alpha, R_plot_ext, T_plot_ext, r"\alpha")
plot_heatmap_variable_r_preciso(err_m, R_plot_ext, T_plot_ext, r"m")
plot_heatmap_variable_r_preciso(err_rho0, R_plot_ext, T_plot_ext, r"\rho_0")
plot_heatmap_variable_r_preciso(err_v, R_plot_ext, T_plot_ext, r"v")

print(R_plot_ext.shape, T_plot_ext.shape, err_alpha.shape)
