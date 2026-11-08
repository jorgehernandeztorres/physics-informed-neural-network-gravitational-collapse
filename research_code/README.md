# Archived research scripts

These scripts are the surviving Python files from the Physics Bachelor's Thesis workspace, copied without changing their scientific logic. They contain exploratory experiments and unused helpers.

Important caveats:
- `PINN_OS_4.py` produces `(alpha, m, rho0, v)`; the archived final thesis prediction file records `(alpha, X, rho0, v)`.
- `main_3.py` imports `PINN_OS`, while the surviving model file is named `PINN_OS_4.py`; the original training entrypoint is therefore **not immediately runnable**.
- Auxiliary plotting scripts use different intermediate filenames and conventions. Use `../scripts/evaluate_saved_predictions.py` for checked evaluation of the final thesis predictions instead.
- Training was resource-intensive and this snapshot has not been shown to recreate the final results of the thesis.

The files are preserved for transparency rather than silently rewritten to imply a fully reproducible final training pipeline.
