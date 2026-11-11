# Physics-Informed Neural Network for Gravitational Collapse

**Bachelor's Thesis in Physics (2025)** | Jorge Hernández Torres  
**Python · PyTorch · Physics-Informed Neural Networks · Automatic Differentiation · General Relativity**

This academic research project explores whether a **physics-informed neural network (PINN)** can approximate the **Oppenheimer-Snyder gravitational-collapse model**. The model describes the spherical collapse of an idealized, pressureless star and provides analytical reference solutions against which neural-network predictions can be evaluated.

The work combines analytical modeling, nonlinear physical constraints, neural-network design, differentiable residuals, and numerical evaluation. The original thesis was completed as part of a double degree in Mathematics and Physics at the **Universitat de València**, with a Physics thesis grade of **9.4/10**.

![Comparison between the theoretical solution and a PINN prediction for the lapse](assets/Figure_alpha.png)

*Original project figure: analytical solution (curves) vs. PINN predictions (markers), at different times.*

## Project at a glance

| Component | Approach |
| --- | --- |
| Physical problem | Oppenheimer-Snyder gravitational collapse in spherical symmetry |
| Learning approach | Continuous-time physics-informed neural network |
| Model inputs | Time and radial coordinate |
| Thesis outputs | Lapse `alpha`, metric coefficient `X`, rest-mass density `rho0`, fluid velocity `v` |
| Training objective | Initial and boundary conditions, plus five differential-equation residuals |
| Main tools | Python, PyTorch, NumPy, SciPy; MATLAB for analytical reference data |
| Evaluation | Relative L2 errors and comparison with analytical profiles |

## Motivation and method

Instead of fitting labeled observations alone, a PINN uses **automatic differentiation** to calculate derivatives of its predictions and penalizes violations of physical equations. For the relativistic-collapse problem, the thesis involved:

1. Deriving and implementing the governing physical equations and reference solution.
2. Defining a feed-forward neural network for spatiotemporal fields.
3. Constructing physics residuals alongside initial- and boundary-condition losses.
4. Training and comparing model predictions with the analytical solution.
5. Identifying numerical and computational limitations.

The thesis describes a four-hidden-layer network (100 units per layer), SiLU activations, and staged Adam/L-BFGS-B optimization. The original thesis-output formulation `(alpha, X, rho0, v)` has since been recovered and is preserved in [`research_code/thesis_formulation/`](research_code/thesis_formulation/). A separate experimental formulation `(alpha, m, rho0, v)` is also retained for research provenance; see [Code provenance and limitations](#code-provenance-and-limitations).

## Results: strengths and limitations

The final archived prediction array matches the relative L2 errors reported in the thesis:

| Physical quantity | Relative L2 error | Interpretation |
| --- | ---: | --- |
| Lapse `alpha` | **0.2363** | Captures broad behavior, with noticeable discrepancies |
| Metric coefficient `X` | **0.1505** | Lowest relative error among the reported fields |
| Rest-mass density `rho0` | **10.9334** | Large errors in strongly nonlinear regions |
| Fluid velocity `v` | **0.9997** | Poor quantitative accuracy near extreme collapse |

As an additional sanity check, the constant baseline `X = 1` achieves a relative L2 error of approximately 0.0865 on the same reference grid, outperforming the PINN's result for `X` (0.1505). This highlights the importance of evaluating scientific ML predictions against simple baselines rather than relying only on relative errors between modeled quantities.

These values are **not** evidence that the model solves gravitational collapse to high accuracy. In particular, density and velocity remain poorly approximated. The thesis reports a substantial computational cost (about 4,995 seconds for its final L-BFGS-B stage in a Colab environment equipped with a T4 GPU). The archived script does not establish whether GPU acceleration was actually used. The project was valuable as an investigation of PINN design and failure modes, not as a faster or more accurate replacement for established solvers.

| Mass profile | Density error |
| --- | --- |
| ![Theoretical and predicted mass profile](assets/Figure_m.png) | ![Absolute error heatmap of density](assets/Error_absoluto_rho0.png) |

### Reproduce the numerical evaluation

The archived reference and final predictions are included in the repository. No GPU or PyTorch installation is necessary to validate the **saved predictions**.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/evaluate_saved_predictions.py
```

Expected output (rounded):

```text
alpha   0.236276
X       0.150524
rho0   10.9334
v       0.999722
```

To create new comparison plots from these stored arrays:

```bash
python scripts/evaluate_saved_predictions.py --plot
```

The command saves figures in `outputs/` (ignored by Git). To run the automated numerical checks:

```bash
python -m unittest discover -s tests -v
```

### Check the recovered thesis model

The original thesis-output implementation can also be checked with a **small one-step PyTorch smoke test** (not a reproduction of the final training):

```bash
python -m pip install -r requirements-training.txt
python scripts/smoke_test_thesis_formulation.py
```

This check requires the original files under `research_code/thesis_formulation/`. The historical `main.py` is preserved for research provenance and is **not** a ready-to-run command in the reorganized repository.

## Repository structure

```text
assets/                  Original thesis result figures
data/reference/          Analytical reference data (MATLAB .mat)
data/predictions/        Final saved PINN outputs evaluated in the thesis
scripts/                 Reproducible evaluation, plotting and model checks
research_code/           Surviving Python experiment scripts, preserved as supplied
matlab/                  Reference-solution generation scripts, preserved as supplied
tests/                   Validation of the saved predictions
requirements.txt         Lightweight dependencies for evaluation
requirements-training.txt  Additional PyTorch dependency for model inspection
```

## Code provenance and limitations

The original thesis-output formulation `(alpha, X, rho0, v)` has been recovered from the 2025 research workspace and is preserved in [`research_code/thesis_formulation/`](research_code/thesis_formulation/).

The recovered training script references the same prediction filename as the archived final results. However, the original trained weights, complete training logs, and exact software environment are unavailable. Therefore, an exact end-to-end reproduction of the final training has not been verified.

A [lightweight smoke test](scripts/smoke_test_thesis_formulation.py) successfully checks the recovered model's physics-based loss, gradient computation, and one Adam optimization step. This does not reproduce the original training results.

A separate experimental formulation `(alpha, m, rho0, v)` is preserved in `research_code/`. Its training scripts and output parametrization differ from the thesis formulation and should not be treated as interchangeable.

The reproducible scientific results currently provided are the evaluation metrics and plots computed from the archived final predictions and analytical reference data. The MATLAB reference-generation scripts are also preserved, but their complete execution has not been independently revalidated. The recovered historical training code also contains known sampling inconsistencies, documented in `research_code/thesis_formulation/README.md`; it should not be presented as a validated reproduction of the physical domain.

For details, see [`research_code/README.md`](research_code/README.md) and [`research_code/thesis_formulation/README.md`](research_code/thesis_formulation/README.md).

## What I learned

- Translating relativistic conservation and metric equations into differentiable optimization objectives.
- Implementing a custom PINN architecture and balancing multiple competing losses.
- Validating scientific ML models against independent analytical solutions.
- Recognizing model failure modes near sharp gradients and relativistic limits.
- Weighing model quality against substantial computational training cost.

**Context:** Academic research project (2025); not a production or commercial deployment. No third-party dataset from a commercial organization is included.
