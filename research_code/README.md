# Archived research code

The `research_code/` folder documents two different historical PINN formulations from the Physics Bachelor's Thesis workspace. All files copied from the 2025 workspace retain their original scientific logic and comments.

| Source | Outputs | Status |
| --- | --- | --- |
| [`thesis_formulation/PINN_OS.py`](thesis_formulation/PINN_OS.py) and [`thesis_formulation/main.py`](thesis_formulation/main.py) | `(alpha, X, rho0, v)` | **Recovered**, matching the thesis-output variables and archived prediction filename; a one-step physics-loss check works. Full training not verified. |
| `PINN_OS_4.py` and `main_3.py` | `(alpha, m, rho0, v)` | Alternative/experimental model. The archived launcher `main_3.py` imports `PINN_OS`, not the corresponding variant file, so it is **not a working top-level entrypoint**. |

The plot/data helper scripts here refer to legacy working-directory paths and filenames. They are preserved as historical artifacts, not runnable in the current layout. For a verified evaluation of the final saved outputs, use `../scripts/evaluate_saved_predictions.py`.

**Never rename one model to `PINN_OS.py` inside this directory just to satisfy the other launcher's import:** the constructor signatures and physical outputs differ. Use the separate checked smoke scripts in `../scripts/` for both formulations. See [`thesis_formulation/README.md`](thesis_formulation/README.md) for detailed limitations of the recovered final-output code.
