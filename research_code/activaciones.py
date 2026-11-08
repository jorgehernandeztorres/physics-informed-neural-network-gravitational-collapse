# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Definición de funciones de activación
def silu(x):
    return x / (1 + np.exp(-x))  # también conocida como swish

def softplus(x):
    return np.log1p(np.exp(x))  # log(1 + e^x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Rango de valores
x = np.linspace(-10, 10, 400)

# Evaluación de las funciones
y_silu = silu(x)
y_softplus = softplus(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)

# Crear el gráfico
plt.figure(figsize=(5, 3))
plt.plot(x, y_sigmoid, color='blue', label='Sigmoide: $\sigma(x) = \\frac{1}{1 + e^{-x}}$')
plt.plot(x, y_silu, color='green', label='SiLU: $x \\cdot \\sigma(x) = \\frac{x}{1 + e^{-x}}$')
plt.plot(x, y_softplus, color='red', label='Softplus: $\\log(1 + e^{x})$')
plt.plot(x, y_tanh, color='purple', label='Tanh: $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$')

# Detalles del gráfico
plt.title('Funciones de Activación Neuronal')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()

# Guardar la imagen
#plt.savefig("funciones_activacion_neuronal_final.png")
plt.show()