# Continuous-time PINN for Oppenheimer-Snyder collapse using PyTorch

import torch
import torch.nn as nn
import numpy as np
import time
# import scipy.io
from scipy.optimize import minimize

# PARÁMETROS DE OS.mat: Nt = 51, Nr = 500, M = 1, R0 = 10, t_max = 50

np.random.seed(1234)
torch.manual_seed(1234)

class NN_arch(nn.Module): # define la arquitectura de la red neuronal (capas, activaciones, normalización)
    def __init__(self, layers, lb, ub):

        super(NN_arch, self).__init__() #Cuando defines una clase que hereda de otra (aquí NN_arch hereda de nn.Module), necesitas asegurarte de que el constructor de la clase base
        # se llama. Esta línea llama al constructor de nn.Module.
        # Inicializa todo lo que PyTorch necesita para que tu clase personalizada funcione como una red neuronal (gestión de parámetros, registro de módulos, etc.)

        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)
        self.activation = nn.SiLU() # funciona mejor que la de tangente hiperbólica, al menos para la matriz pequeña y red neuronal pequeña
        self.layers = layers

        self.linears = nn.ModuleList() # nn.ModuleList() es una colección registrada de capas (o submódulos) dentro de tu modelo. Se comporta como una lista de Python, pero con una
        # diferencia clave: PyTorch sabe que todos los elementos dentro de una ModuleList son parte del modelo, y los incluye automáticamente en .parameters() (para entrenamiento),
        # .to(device) (para mover a GPU/CPU), .eval() / .train() (para cambiar el modo del modelo). Luego ModuleList garantiza que PyTorch registre correctamente todas esas capas

        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i], layers[i+1])

            # Las siguientes dos líneas controlan cómo se inicializan los pesos y los sesgos (biases) de las capas lineales en tu red neuronal:

            #nn.init.xavier_normal_(linear.weight) # Inicializa los pesos de la capa linear usando la distribución normal de Xavier (Glorot). Esta inicialización es adecuada cuando usas
            # activaciones simétricas no lineales, como Tanh o Sigmoid. Se basa en la idea de mantener la varianza del gradiente constante a lo largo de las capas durante el backpropagation.
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu') # Para activaciones no simétricas como SiLU, la literatura recomienda usar inicialización de He (He-et-al / Kaiming).
            # Aunque no estás usando ReLU, usar 'relu' como argumento para nonlinearity se acepta comúnmente para SiLU, porque proporciona una varianza de salida adecuada para capas
            # profundas

            nn.init.zeros_(linear.bias) # Inicializa el bias de la capa a cero. Es seguro y común, ya que el sesgo se aprende durante el entrenamiento.
            # ¿Por qué es buena práctica?
            # Evita gradientes explosivos o desvanecientes al inicio del entrenamiento.
            # Mejora la convergencia de redes profundas.
            # Xavier es más adecuado que inicialización aleatoria sin control, sobre todo con activaciones como Tanh.

            self.linears.append(linear)

    def scale_input(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, x):
        H = self.scale_input(x)
        for linear in self.linears[:-1]: # self.linears[:-1] significa: todas las capas menos la última
            H = self.activation(linear(H))
        out = self.linears[-1](H)
        # Activaciones seguras + codificación inductiva o "physics-informed regularization" ; QUIZÁS MEJOR PROBAR A METER SALIDAS NORMALES (sin activación)
        alpha = torch.sigmoid(torch.clamp(out[:,0:1], -40.0, 40.0))          # SI SALE MAL, volver a torch.sigmoid(torch.clamp(out[:,0:1], min=-40.0, max=40.0))
        X = 1.0 + torch.nn.functional.softplus(out[:,1:2])                   # garantiza X > 1
        rho0 = torch.nn.functional.softplus(out[:,2:3])                      # garantiza rho0 > 0
        v = 0.5*torch.tanh(torch.clamp(out[:,3:4], -40.0, 40.0)) - 0.5
        return torch.cat([alpha, X, rho0, v], dim=1)


class PhysicsInformedModel: # gestiona la lógica del entrenamiento físico, cálculo de pérdidas, optimización, etc.
    def __init__(self, model, r0, tb, Rtb, Xc, M):

        # r0: Vector de posiciones en el instante inicial
        
        r0 = r0.reshape(-1, 1)  # ← asegura shape (N0, 1)
        X0 = np.concatenate((np.zeros_like(r0), r0), axis=1) # (0, r0)

        tb = tb.reshape(-1, 1)  # ← asegura shape (Nb, 1)
        Rtb = Rtb.reshape(-1, 1)  # ← asegura shape (Nb, 1)
        Xb0 = np.concatenate((tb, np.zeros_like(tb)), axis=1) # (tb, 0)
        XbR = np.concatenate((tb, Rtb), axis=1) # (tb, R(tb))

        self.X0 = torch.tensor(X0, dtype=torch.float32, requires_grad=True)
        self.Xb0 = torch.tensor(Xb0, dtype=torch.float32, requires_grad=True)
        self.XbR = torch.tensor(XbR, dtype=torch.float32, requires_grad=True)
        self.Xc = torch.tensor(Xc, dtype=torch.float32, requires_grad=True)

        self.M = M

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # # Inicialización de SoftAdapt
        # self.prev_losses = {'ic': 1.0, 'bc': 1.0, 'pde': 1.0}
        # self.weights = {'ic': 1.0, 'bc': 1.0, 'pde': 1.0}
        # self.softadapt_beta = 0.1  # Puedes probar con 0.1, 0.5 o 1.0 según sensibilidad

    def net_outputs(self, x):
        output = self.model(x)
        return output[:,0:1], output[:,1:2], output[:,2:3], output[:,3:4]

    def compute_residuals(self, x): # en x yo espero introducir los valores tanto de t como de r en columnas, metiendo primero los de t y luego los de r
        r = x[:,1:2]
        alpha, X, rho0, v = self.net_outputs(x)
        
        print("Alpha stats:", torch.min(alpha), torch.max(alpha))
        print("X stats:", torch.min(X), torch.max(X))
        print("rho0 stats:", torch.min(rho0), torch.max(rho0))
        print("v stats:", torch.min(v), torch.max(v))

        m = 0.5 * r * (1 - 1 / X**2)
        Phi = torch.log(alpha)
        W = 1 / torch.sqrt(1 - v**2)
        D = X * rho0 * W
        S = rho0 * W**2 * v
        tau = rho0 * W**2 - D

        # multiplico las ecuaciones de las variables de conservación por r^2 para evitarme problemas de dividir por un número muy pequeño

        D_t = torch.autograd.grad(D, x, grad_outputs=torch.ones_like(D), retain_graph=True, create_graph=True)[0][:, 0:1]
        S_t = torch.autograd.grad(S, x, grad_outputs=torch.ones_like(S), retain_graph=True, create_graph=True)[0][:, 0:1]
        tau_t = torch.autograd.grad(tau, x, grad_outputs=torch.ones_like(tau), retain_graph=True, create_graph=True)[0][:, 0:1]

        f1 = D*v
        f2 = S*v
        f3 = S - D*v

        factor = r**2 * alpha / X
        f1_r = torch.autograd.grad(factor*f1, x, grad_outputs=torch.ones_like(f1), retain_graph=True, create_graph=True)[0][:, 1:2]
        f2_r = torch.autograd.grad(factor*f2, x, grad_outputs=torch.ones_like(f2), retain_graph=True, create_graph=True)[0][:, 1:2]
        f3_r = torch.autograd.grad(factor*f3, x, grad_outputs=torch.ones_like(f3), retain_graph=True, create_graph=True)[0][:, 1:2]

        s2 = (S*v - tau - D) * alpha * X * m

        R1 = r**2 * D_t + f1_r
        R2 = r**2 * S_t + f2_r - s2
        R3 = r**2 * tau_t + f3_r

        # también multiplico la ecuación de Phi por r^2

        m_r = torch.autograd.grad(m, x, grad_outputs=torch.ones_like(m), retain_graph=True, create_graph=True)[0][:, 1:2]
        Phi_r = torch.autograd.grad(Phi, x, grad_outputs=torch.ones_like(Phi), retain_graph=True, create_graph=True)[0][:, 1:2]

        R4 = m_r - 4 * np.pi * r**2 * (D + tau)
        R5 = r**2 * Phi_r - X**2 * (m + 4 * np.pi * r**3 * S * v)

        return R1, R2, R3, R4, R5

    # def update_softadapt_weights(self, curr_losses):
    #     delta_ic = self.prev_losses['ic'] - curr_losses['ic']
    #     delta_bc = self.prev_losses['bc'] - curr_losses['bc']
    #     delta_pde = self.prev_losses['pde'] - curr_losses['pde']

    #     deltas = torch.tensor([
    #         max(0.0, delta_ic),
    #         max(0.0, delta_bc),
    #         max(0.0, delta_pde)
    #     ])

    #     weights = torch.softmax(deltas / self.softadapt_beta, dim=0)

    #     self.weights['ic'] = weights[0].item()
    #     self.weights['bc'] = weights[1].item()
    #     self.weights['pde'] = weights[2].item()

    #     self.prev_losses = curr_losses.copy()

    def loss_ic(self):
        _, X_0, rho0_0, v_0 = self.net_outputs(self.X0)
        R0 = self.X0[-1,1:2]
        r0 = self.X0[:,1:2]
        m_0 = 0.5 * r0 * (1 - 1 / X_0**2)
        m_0_true = self.M * (r0 / R0)**3
        rho0_0_true =  3 * self.M / (4 * np.pi * R0**3)
        eps = 1e-2

        # Calcular errores individuales (como MSE o L2)
        loss_m_0 = torch.mean((m_0 - m_0_true) ** 2) / torch.mean(m_0_true ** 2) + (X_0[0,0:1] - 1) ** 2 # añado este sumando porque, por construcción,
        # m_0(r=0)=0=m_0_true(r=0), con lo que ahí desaparece cualquier penalización que pueda existir sobre X_0(r=0)
        loss_rho0_0 = torch.mean((rho0_0 - rho0_0_true) ** 2) / (rho0_0_true ** 2)
        loss_v_0 = torch.mean(v_0 ** 2) / eps

        # # Encontrar el máximo error
        # ic_max = torch.mean(torch.stack([torch.mean(torch.abs(m_0_cor)), torch.mean(torch.abs(rho0_0_cor)), torch.mean(torch.abs(v_0))], dim=0))

        return (loss_m_0 + loss_rho0_0 + loss_v_0) #/ ic_max

    def loss_bc(self):
        # r=0 boundary
        _, X_b0, _, v_b0 = self.net_outputs(self.Xb0)
        # rb0 = self.Xb0[:,1:2] = vector de 0's
        # m_b0 = 0.5 * rb0 * (1 - 1 / X_b0**2) # = 0 siempre --> no hace falta ponerlo
        eps = 1e-2
        loss_X_b0 = torch.mean((X_b0 - 1) ** 2)
        loss_v_b0 = torch.mean(v_b0 ** 2) / eps

        # r=R0 boundary
        _, X_bR, _, _ = self.net_outputs(self.XbR)
        rbR = self.XbR[:,1:2]
        m_bR = 0.5 * rbR * (1 - 1 / X_bR**2)
        m_bR_true = self.M
        loss_m_bR = torch.mean((m_bR - m_bR_true) **2) / (m_bR_true ** 2)

        # # Encontrar el máximo error
        # bc_max = torch.mean(torch.stack([torch.mean(torch.abs(m_b0)), torch.mean(torch.abs(m_bR_cor)), torch.mean(torch.abs(v_b0)), torch.mean(torch.abs(v_bR))], dim=0))

        return (loss_X_b0 + loss_v_b0 + loss_m_bR) #/ bc_max

    def loss_pde(self):
        R1, R2, R3, R4, R5 = self.compute_residuals(self.Xc)
        eps = 1e-2

        loss_R1 = torch.mean(R1**2) / eps
        loss_R2 = torch.mean(R2**2) / eps
        loss_R3 = torch.mean(R3**2) / eps
        loss_R4 = torch.mean(R4**2) / eps
        loss_R5 = torch.mean(R5**2) / eps

        # # Encontrar el máximo error
        # resid_max = torch.mean(torch.stack([torch.mean(torch.abs(R1)), torch.mean(torch.abs(R2)), torch.mean(torch.abs(R3)), torch.mean(torch.abs(R4)), torch.mean(torch.abs(R5))], dim=0))

        return (loss_R1 + loss_R2 + loss_R3 + loss_R4 + loss_R5) #/ resid_max

    def loss(self): # puedo ir variando los pesos de ambas loss para ver de qué forma se obtienen mejores resultados
        weight_ic = 1.0
        weight_bc = 1.0
        weight_pde = 10.0
        return weight_ic*self.loss_ic() + weight_bc*self.loss_bc() + weight_pde*self.loss_pde()

        # # Cálculo de pérdidas individuales
        # loss_ic_val = self.loss_ic()
        # loss_bc_val = self.loss_bc()
        # loss_pde_val = self.loss_pde()

        # # Almacenar pérdidas actuales
        # curr_losses = {
        #     'ic': loss_ic_val.item(),
        #     'bc': loss_bc_val.item(),
        #     'pde': loss_pde_val.item()
        #     }

        # # Actualizar pesos dinámicos
        # self.update_softadapt_weights(curr_losses)

        # return self.weights['ic'] * loss_ic_val + self.weights['bc'] * loss_bc_val + self.weights['pde'] * loss_pde_val

    def get_params(self):
        return torch.cat([p.flatten() for p in self.model.parameters()]).detach().numpy()

    def set_params(self, flat_params): # Espera recibir un torch.Tensor
        pointer = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
            pointer += numel

    def loss_and_grad(self, flat_params): # Espera recibir un np.ndarray
        tensor_params = torch.tensor(flat_params, dtype=torch.float32, requires_grad=True)
        self.set_params(tensor_params)
        self.model.zero_grad()
        loss_val = self.loss() # esto no quiere decir que la red neuronal haya actuado sobre unos ciertos inputs y haya dado una respuesta, sino que simplemente esto llama a la función
        # loss. Luego para realmente usar esto luego tendremos que hacer algo como self.loss_and_grad(torch.tensor(w, requires_grad=True))
        loss_val.backward()
        # Removed torch.autograd.set_detect_anomaly(True) for performance
        grad = torch.cat([p.grad.flatten() for p in self.model.parameters()]).detach().numpy()

        return loss_val.item(), grad

    def callback(self, flat_params): # Espera recibir un np.ndarray
        tensor_params = torch.tensor(flat_params, dtype=torch.float32)
        self.set_params(tensor_params)
        loss_val = self.loss().item()
        print(f"L-BFGS-B callback: loss = {loss_val:.3e}")

    def train_adam(self, epochs):
        start_time = time.time()
        for ep in range(epochs):
            self.optimizer.zero_grad()
            loss_val = self.loss()
            if ep == 0:
                elapsed = time.time() - start_time
                print(f"Adam Epoch {ep+1}: Loss = {loss_val.item():.3e}, Elapsed time = {elapsed:.2f}, Loss ic = {self.loss_ic().item():.3e}, Loss bc = {self.loss_bc().item():.3e}, Loss pde = {self.loss_pde().item():.3e}")
                start_time = time.time()
            if (ep+1) % 100 == 0: # Modificar a % 500 cuando aumente el número de epochs
                elapsed = time.time() - start_time
                print(f"Adam Epoch {ep+1}: Loss = {loss_val.item():.3e}, Elapsed time = {elapsed:.2f}, Loss ic = {self.loss_ic().item():.3e}, Loss bc = {self.loss_bc().item():.3e}, Loss pde = {self.loss_pde().item():.3e}")
                start_time = time.time()
            loss_val.backward()
            # Removed torch.autograd.set_detect_anomaly(True) for performance
            self.optimizer.step()

    def train_lbfgs(self):
        start_time = time.time()
        print("\nStarting L-BFGS-B optimization...")
        init_params = self.get_params()
        result = minimize(fun=self.loss_and_grad, # scipy.optimize.minimize va a pasar un np.ndarray como argumento a loss_and_grad, y tú necesitas un torch.Tensor(requires_grad=True);
                          # de ahí que en loss_and_grad se introduzca el .requires_grad_(True)
                          x0=init_params,
                          jac=True,
                          method='L-BFGS-B',
                          callback=self.callback,
                          options={'maxiter': 5000, 'maxfun': 50000, 'maxcor': 50, 'ftol': 1.0 * np.finfo(float).eps})
        print(result.message)
        elapsed = time.time() - start_time
        final_params = torch.tensor(result.x, dtype=torch.float32)
        self.set_params(final_params)
        print(f"L-BFGS-B optimization finished. Elapsed time = {elapsed:.2f}")

    def train(self, epochs):
        start_time = time.time()
        self.train_adam(epochs)
        self.train_lbfgs()
        elapsed = time.time() - start_time
        print((f"Total elapsed time = {elapsed:.2f}"))

    def predict(self, X_star):
        self.model.eval()
        X_star_tensor = torch.tensor(X_star, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_star_tensor).detach().numpy() # Aunque la capa input de la red neuronal espera 2 variables, no es necesario separar el input (que es un array 2D con 2
             # columnas) y dar cada columna por separado. Está bien como lo hago actualmente al pasar una matriz de entrada con 2 columnas.