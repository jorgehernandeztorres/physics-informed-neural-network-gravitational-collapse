# Recovered thesis-output formulation (historical source)

This folder contains **unchanged copies** of `PINN_OS.py` and `main.py` recovered from the original Physics Bachelor's Thesis workspace.

- `PINN_OS.py` defines a PyTorch network with outputs **`(alpha, X, rho0, v)`**, corresponding to the variables discussed in the final thesis. It defines the five physics residuals, initial/boundary losses, Adam optimization and an L-BFGS-B stage.
- `main.py` builds the four-hidden-layer, 100-unit network, samples initial and collocation points, runs the optimizer, evaluates predictions on the analytical grid and writes `results/OS_pred_def_de_verdad_de_la_buena.npz`.
- That named `.npz` file in the original workspace is **byte-for-byte identical** to `../../data/predictions/final_thesis_predictions.npz` included in this repository.

## What is and is not verified

The network implementation has been imported and tested with real `OS.mat` values: it can calculate physics residuals, backpropagate and take **one Adam training step**. Run the checked, short test at the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-training.txt
python scripts/smoke_test_thesis_formulation.py
```

**The original `main.py` is preserved for provenance, not offered as a working top-level command.** It expects the historical working directories (`./main/Data/OS.mat` and `./results/`) and the optional `scikit-optimize` (`skopt`) package; those paths no longer exist in the same layout. Its defaults also imply an expensive full training run. If studying the original script, install `scikit-optimize` separately, then review its path assumptions before running it. Do not run the original 1,000-epoch + L-BFGS-B experiment unintentionally.

**Scientific provenance caveat:** the saved prediction file's name matches the output target of `main.py`, but the archived files do not include final trained model weights, a training log or an exact environment snapshot, so we cannot independently verify that this exact source and sampled points generated the saved predictions. In particular, `main.py` samples collocation points over the bounding rectangle `[0, T] x [0, R0]`, whereas the thesis describes the evolving stellar interior `r <= R(t)`. A second caution: the initial-condition loss in `PINN_OS.py` uses the **last supplied sampled initial radius** as `R0`, whereas `main.py` samples the initial radii via Latin hypercube sampling without ensuring the last sampled radius is the true stellar initial radius. This can change the intended initial-condition target. Both differences need investigation before claiming precise end-to-end rerun fidelity. The analytical reference grid is stored in `../../data/reference/OS.mat`.

The separate `../PINN_OS_4.py` / `../main_3.py` files relate to another experimental parametrization with outputs `(alpha, m, rho0, v)`; they are **not interchangeable** with the recovered `(alpha, X, rho0, v)` implementation. Do not move these files together or fix an import by renaming one model to the other.

The independently reproduced scientific results in this repository remain the **relative errors and plots computed from archived final predictions**, not a reproduction of the full training trajectory.
