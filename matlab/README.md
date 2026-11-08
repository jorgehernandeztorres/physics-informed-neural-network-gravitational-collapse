# Analytical reference data (MATLAB)

The original scripts used to construct the Oppenheimer-Snyder analytical reference solution are preserved here. `sol_anal_OS.m` is the main generating script; it calls auxiliary MATLAB functions in this directory and saves the resulting matrices in an `OS.mat` file.

The reference grid used for evaluation is supplied separately at `../data/reference/OS.mat`. MATLAB reconstruction has not been rerun as part of this GitHub packaging step.
