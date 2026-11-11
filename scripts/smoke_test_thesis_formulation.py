#!/usr/bin/env python3
"""Check that the recovered (alpha, X, rho0, v) thesis-source PINN can train.

This is a one-step CPU smoke test on tiny samples of the original reference grid,
NOT an attempt to reproduce the 2025 thesis's full training or final metrics.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "research_code" / "thesis_formulation" / "PINN_OS.py"


def main():
    spec = importlib.util.spec_from_file_location("recovered_thesis_model", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load recovered thesis model")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = loadmat(ROOT / "data" / "reference" / "OS.mat")
    t = raw["t_vector"].ravel()
    r = raw["r_matrix"]
    tmax = float(t[-1])
    R0 = float(r[0, -1])

    # All collocation coordinates come from the analytical interior grid.
    rng = np.random.default_rng(2025)
    time_indices = rng.integers(1, len(t), size=16)
    radius_indices = rng.integers(1, r.shape[1] - 1, size=16)
    xc = np.column_stack((t[time_indices], r[time_indices, radius_indices]))
    initial_radii = np.linspace(0.0, R0, 10)
    boundary_times = t[[1, 10, 20, 35, 50]]
    boundary_radii = r[[1, 10, 20, 35, 50], -1]

    network = mod.NN_arch([2, 8, 8, 4],
                          np.array([0.0, 0.0]), np.array([tmax, R0]))
    pinn = mod.PhysicsInformedModel(
        network, initial_radii, boundary_times, boundary_radii, xc, M=1.0
    )
    initial_loss = float(pinn.loss().detach().cpu().item())
    if not np.isfinite(initial_loss):
        raise RuntimeError(f"Non-finite initial loss: {initial_loss}")
    pinn.train_adam(1)
    after_loss = float(pinn.loss().detach().cpu().item())
    if not np.isfinite(after_loss):
        raise RuntimeError(f"Non-finite loss after one step: {after_loss}")
    if not all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in network.parameters()):
        raise RuntimeError("Missing or non-finite gradients")
    sample = network(torch.tensor(xc[:3], dtype=torch.float32))
    if sample.shape != (3, 4) or not torch.isfinite(sample).all():
        raise RuntimeError("Unexpected output dimensions or values")
    print(f"Recovered thesis-formulation smoke test PASSED (Adam: 1 step; "
          f"initial loss={initial_loss:.4g}, loss after step={after_loss:.4g}).")
    print("NOTE: this does not reproduce the original full training or final metrics.")


if __name__ == "__main__":
    main()
