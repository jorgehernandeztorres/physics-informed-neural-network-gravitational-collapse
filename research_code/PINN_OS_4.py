# Continuous-time PINN for Oppenheimer-Snyder collapse using PyTorch (CUDA Optimized)

import torch
import torch.nn as nn
import numpy as np
import time
from scipy.optimize import minimize

# === Selección de dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# PARÁMETROS DE OS.mat: Nt = 51, Nr = 500, M = 1, R0 = 10, t_max = 50
np.random.seed(1234)
torch.manual_seed(1234)

class NN_arch(nn.Module):
    def __init__(self, layers, lb, ub, M, R0):
        super(NN_arch, self).__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.activation = nn.SiLU()
        self.layers = layers
        self.M = M
        self.R0 = R0
        self.linears = nn.ModuleList()
        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i], layers[i+1])
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)

    def scale_input(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, x):
        H = self.scale_input(x)
        for linear in self.linears[:-1]:
            H = self.activation(linear(H))
        out = self.linears[-1](H)
        alpha = torch.sigmoid(torch.clamp(out[:,0:1], -40.0, 40.0))
        m = self.M * torch.sigmoid(torch.clamp(out[:,1:2], -40.0, 40.0))
        rho0 = 0.03 * torch.sigmoid(torch.clamp(out[:,2:3], -40.0, 40.0)) + 3 * self.M / (4 * np.pi * self.R0**3)
        v = torch.sigmoid(torch.clamp(out[:,3:4], -40.0, 40.0)) - 1.0
        return torch.cat([alpha, m, rho0, v], dim=1)

class PhysicsInformedModel:
    def __init__(self, model, r0, tb, Rtb, Xc, M):
        r0 = r0.reshape(-1, 1)
        X0 = np.concatenate((np.zeros_like(r0), r0), axis=1)
        tb = tb.reshape(-1, 1)
        Rtb = Rtb.reshape(-1, 1)
        Xb0 = np.concatenate((tb, np.zeros_like(tb)), axis=1)
        XbR = np.concatenate((tb, Rtb), axis=1)
        self.X0 = torch.tensor(X0, dtype=torch.float32, requires_grad=True).to(device)
        self.Xb0 = torch.tensor(Xb0, dtype=torch.float32, requires_grad=True).to(device)
        self.XbR = torch.tensor(XbR, dtype=torch.float32, requires_grad=True).to(device)
        self.Xc = torch.tensor(Xc, dtype=torch.float32, requires_grad=True).to(device)
        self.M = M
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def net_outputs(self, x):
        output = self.model(x)
        return output[:,0:1], output[:,1:2], output[:,2:3], output[:,3:4]

    def compute_residuals(self, x):
        r = x[:,1:2]
        alpha, m, rho0, v = self.net_outputs(x)
        print("Alpha stats:", torch.min(alpha), torch.max(alpha))
        print("m stats:", torch.min(m), torch.max(m))
        print("rho0 stats:", torch.min(rho0), torch.max(rho0))
        print("v stats:", torch.min(v), torch.max(v))
        eps = torch.tensor(1e-6, device=device)
        X = 1 / torch.sqrt(torch.clamp(1 - 2*m/(r+eps), min=eps))
        Phi = torch.log(torch.clamp(alpha, min=eps))
        W = 1 / torch.sqrt(torch.clamp(1 - v**2, min=eps))
        D = X * rho0 * W
        S = rho0 * W**2 * v
        tau = rho0 * W**2 - D
        D_t = torch.autograd.grad(D, x, grad_outputs=torch.ones_like(D).to(device), retain_graph=True, create_graph=True)[0][:, 0:1]
        S_t = torch.autograd.grad(S, x, grad_outputs=torch.ones_like(S).to(device), retain_graph=True, create_graph=True)[0][:, 0:1]
        tau_t = torch.autograd.grad(tau, x, grad_outputs=torch.ones_like(tau).to(device), retain_graph=True, create_graph=True)[0][:, 0:1]
        f1 = D*v
        f2 = S*v
        f3 = S - D*v
        factor = r**2 * alpha / X
        f1_r = torch.autograd.grad(factor*f1, x, grad_outputs=torch.ones_like(f1).to(device), retain_graph=True, create_graph=True)[0][:, 1:2]
        f2_r = torch.autograd.grad(factor*f2, x, grad_outputs=torch.ones_like(f2).to(device), retain_graph=True, create_graph=True)[0][:, 1:2]
        f3_r = torch.autograd.grad(factor*f3, x, grad_outputs=torch.ones_like(f3).to(device), retain_graph=True, create_graph=True)[0][:, 1:2]
        s2 = (S*v - tau - D) * alpha * X * m / (r**2 + eps)
        R1 = r**2 * D_t + f1_r
        R2 = r**2 * S_t + f2_r - s2
        R3 = r**2 * tau_t + f3_r
        m_r = torch.autograd.grad(m, x, grad_outputs=torch.ones_like(m).to(device), retain_graph=True, create_graph=True)[0][:, 1:2]
        Phi_r = torch.autograd.grad(Phi, x, grad_outputs=torch.ones_like(Phi).to(device), retain_graph=True, create_graph=True)[0][:, 1:2]
        R4 = m_r - 4 * np.pi * r**2 * (D + tau)
        R5 = r**2 * Phi_r - X**2 * (m + 4 * np.pi * r**3 * S * v)
        return R1, R2, R3, R4, R5

    def loss_ic(self):
        _, m_0, rho0_0, v_0 = self.net_outputs(self.X0)
        R0 = self.X0[-1,-1]
        r0 = self.X0[:,1:2]
        m_0_true = self.M * (r0 / R0)**3
        rho0_0_true = 3 * self.M / (4 * np.pi * R0**3)
        eps = torch.tensor(1e-2, device=device)
        loss_m_0 = torch.mean((m_0 - m_0_true) ** 2) / torch.mean(m_0_true ** 2)
        loss_rho0_0 = torch.mean((rho0_0 - rho0_0_true) ** 2) / (rho0_0_true ** 2)
        loss_v_0 = torch.mean(v_0 ** 2) / eps
        return (loss_m_0 + 1e-2*loss_rho0_0 + loss_v_0)

    def loss_bc(self):
        _, m_b0, _, v_b0 = self.net_outputs(self.Xb0)
        eps = torch.tensor(1e-2, device=device)
        loss_m_b0 = torch.mean(m_b0 ** 2) / eps
        loss_v_b0 = torch.mean(v_b0 ** 2) / eps
        _, m_bR, _, _ = self.net_outputs(self.XbR)
        m_bR_true = self.M
        loss_m_bR = torch.mean((m_bR - m_bR_true) **2) / (m_bR_true ** 2)
        return (loss_m_b0 + loss_v_b0 + loss_m_bR)

    def loss_pde(self):
        R1, R2, R3, R4, R5 = self.compute_residuals(self.Xc)
        eps = torch.tensor(1e-2, device=device)
        loss_R1 = torch.mean(R1**2) / eps
        loss_R2 = torch.mean(R2**2) / eps
        loss_R3 = torch.mean(R3**2) / eps
        loss_R4 = torch.mean(R4**2) / eps
        loss_R5 = torch.mean(R5**2) / eps
        print(f"Loss: R1={loss_R1:.3e}, R2={loss_R2:.3e}, R3={loss_R3:.3e}, R4={loss_R4:.3e}, R5={loss_R5:.3e}")
        return (1e2*loss_R1 + 0.1*loss_R2 + 1e2*loss_R3 + 1e-2*loss_R4 + 1e-11*loss_R5)

    def loss(self):
        weight_ic = 1.0
        weight_bc = 1.0
        weight_pde = 20.0
        return weight_ic*self.loss_ic() + weight_bc*self.loss_bc() + weight_pde*self.loss_pde()

    def get_params(self):
        return torch.cat([p.flatten() for p in self.model.parameters()]).detach().cpu().numpy()

    def set_params(self, flat_params):
        pointer = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(torch.tensor(flat_params[pointer:pointer + numel], dtype=torch.float32).view_as(p).to(device))
            pointer += numel

    def loss_and_grad(self, flat_params):
        tensor_params = torch.tensor(flat_params, dtype=torch.float32, requires_grad=True).to(device)
        self.set_params(tensor_params)
        self.model.zero_grad()
        loss_val = self.loss()
        loss_val.backward()
        grad = torch.cat([p.grad.flatten() for p in self.model.parameters()]).detach().cpu().numpy()
        return loss_val.item(), grad

    def callback(self, flat_params):
        tensor_params = torch.tensor(flat_params, dtype=torch.float32).to(device)
        self.set_params(tensor_params)
        loss_val = self.loss().item()
        print(f"L-BFGS-B callback: loss = {loss_val:.3e}")

    def train_adam(self, epochs):
        start_time = time.time()
        for ep in range(epochs):
            self.optimizer.zero_grad()
            loss_val = self.loss()
            if ep == 0 or (ep+1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Adam Epoch {ep+1}: Loss = {loss_val.item():.3e}, Elapsed time = {elapsed:.2f}, Loss ic = {self.loss_ic().item():.3e}, Loss bc = {self.loss_bc().item():.3e}, Loss pde = {self.loss_pde().item():.3e}")
                start_time = time.time()
            loss_val.backward()
            self.optimizer.step()

    def train_lbfgs(self):
        start_time = time.time()
        print("\nStarting L-BFGS-B optimization...")
        init_params = self.get_params()
        result = minimize(fun=self.loss_and_grad,
                          x0=init_params,
                          jac=True,
                          method='L-BFGS-B',
                          callback=self.callback,
                          options={'maxiter': 50000, 'maxfun': 50000, 'maxcor': 50, 'ftol': 1e-6})
        print(result.message)
        elapsed = time.time() - start_time
        final_params = torch.tensor(result.x, dtype=torch.float32).to(device)
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
        X_star_tensor = torch.tensor(X_star, dtype=torch.float32).to(device)
        with torch.no_grad():
            return self.model(X_star_tensor).detach().cpu().numpy()