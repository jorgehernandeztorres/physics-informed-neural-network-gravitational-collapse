#!/usr/bin/env python3
"""Re-evaluate saved Physics thesis predictions against the analytical OS grid.

This script evaluates archived outputs; it does not retrain the neural network.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'data/reference/OS.mat'
PREDICTIONS = ROOT / 'data/predictions/final_thesis_predictions.npz'
FIELDS = (('alpha', 'alpha_matrix'), ('X', 'X_matrix'),
          ('rho0', 'rho0_matrix'), ('v', 'v_matrix'))


def load_arrays(reference=REFERENCE, predictions=PREDICTIONS):
    ref = loadmat(reference)
    with np.load(predictions, allow_pickle=False) as f:
        pred = f['U_pred']
    shape = ref['alpha_matrix'].shape
    if pred.shape != (shape[0] * shape[1], 4):
        raise ValueError(f'Expected {(shape[0]*shape[1],4)} but got {pred.shape}')
    if any(ref[k].shape != shape for _,k in FIELDS):
        raise ValueError('Inconsistent analytical grid dimensions')
    if not np.isfinite(pred).all() or any(not np.isfinite(ref[k]).all() for _,k in FIELDS):
        raise ValueError('Non-finite data detected')
    return ref, pred.reshape(shape[0],shape[1],4)


def calculate_relative_errors(reference, prediction):
    errors = {}
    for col,(name,key) in enumerate(FIELDS):
        target = np.asarray(reference[key],dtype=np.float64)
        output = np.asarray(prediction[:,:,col],dtype=np.float64)
        denom = np.linalg.norm(target.ravel())
        if denom == 0:
            raise ValueError(f'Zero reference norm: {name}')
        errors[name] = float(np.linalg.norm((target-output).ravel())/denom)
    return errors


def create_plots(reference,predictions,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True,exist_ok=True)
    t = reference['t_vector'].ravel()
    radii = reference['r_matrix']
    selected = np.linspace(0,radii.shape[0]-1,6,dtype=int)
    for col,(name,key) in enumerate(FIELDS):
        actual = reference[key]
        fig,ax = plt.subplots(figsize=(9,5))
        for idx in selected:
            ax.plot(radii[idx]/2,actual[idx],label=f'Analytical t={t[idx]:g}')
            ax.scatter(radii[idx,::20]/2,predictions[idx,::20,col],marker='x',s=14,label=f'PINN t={t[idx]:g}')
        ax.set(xlabel='r/(2M)',ylabel=name,title=f'Oppenheimer-Snyder: analytical vs PINN ({name})')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7,ncol=2)
        fig.tight_layout()
        fig.savefig(dest/f'comparison_{name}.png',dpi=150)
        plt.close(fig)
        fig,ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(np.abs(actual-predictions[:,:,col]),origin='lower',aspect='auto')
        ax.set(xlabel='Radial grid index',ylabel='Time index',title=f'Absolute error on sampled grid ({name})')
        fig.colorbar(im,ax=ax,label='Absolute error')
        fig.tight_layout()
        fig.savefig(dest/f'absolute_error_{name}.png',dpi=150)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot',action='store_true',help='Save plots to outputs/ (ignored by Git)')
    args=parser.parse_args()
    reference,pred=load_arrays()
    errors=calculate_relative_errors(reference,pred)
    print('Relative L2 error (archived predictions vs analytical solution):')
    for name in ('alpha','X','rho0','v'):
        print(f'{name:<6} {errors[name]:.6g}')
    if args.plot:
        dest=ROOT/'outputs'
        create_plots(reference,pred,dest)
        print(f'Figures saved to {dest}')

if __name__=='__main__':
    main()
