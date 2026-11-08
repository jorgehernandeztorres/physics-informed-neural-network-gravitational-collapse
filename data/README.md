# Data provenance

- `reference/OS.mat`: generated analytical/reference arrays from the author's thesis workspace. Grid: 51 time steps x 500 radial positions.
- `predictions/final_thesis_predictions.npz`: preserved prediction array `U_pred` of shape `(25500, 4)`. Its columns correspond **in order** to `alpha`, `X`, `rho0`, and `v` (not `m`).

The source archives included more than one prediction file. Only the one whose four relative L2 errors matched the final thesis table is included under the explicit `final_thesis_predictions.npz` name. Another archived prediction file belonged to a different experiment and should not be confused with the final thesis result.

Both data files are scientific project artifacts, not personal records or customer data. Reuse or redistribution of this material is not licensed explicitly by this repository.
