#!/usr/bin/env python3
"""Run the surviving experimental PINN variant on the bundled reference grid.

This wrapper fixes the archived launcher's missing module import, but preserves
its physical model and loss functions. It is NOT the final thesis training run.
The default configuration is a tiny CPU smoke run, not meaningful training.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archived_pinn_variant',ROOT/'research_code/PINN_OS_4.py')
variant=importlib.util.module_from_spec(spec)
spec.loader.exec_module(variant)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=1,help='Adam epochs (default: 1 for quick check)')
    parser.add_argument('--collocation',type=int,default=16,help='Collocation points (default: 16)')
    parser.add_argument('--width',type=int,default=16,help='Hidden-layer width')
    parser.add_argument('--hidden-layers',type=int,default=2,help='Hidden-layer count')
    parser.add_argument('--lbfgs',action='store_true',help='Also perform expensive original L-BFGS-B phase')
    parser.add_argument('--save-predictions',action='store_true',help='Save experimental variant predictions to outputs/')
    args=parser.parse_args()
    if min(args.epochs,args.collocation,args.width,args.hidden_layers)<1:
        parser.error('All numeric settings must be positive')
    ref=loadmat(ROOT/'data/reference/OS.mat')
    t=ref['t_vector'].ravel()
    r=ref['r_matrix']
    nt,nr=r.shape
    max_points=(nt-1)*(nr-2)
    if args.collocation>max_points:
        parser.error(f'collocation must be <= {max_points}')
    M=1.0
    R0=float(r[0,-1])
    lower=np.array([0.,0.])
    upper=np.array([float(t[-1]),R0])
    net=variant.NN_arch([2]+[args.width]*args.hidden_layers+[4],lower,upper,M,R0)
    rng=np.random.default_rng(1234)
    i,j=np.meshgrid(np.arange(1,nt),np.arange(1,nr-1),indexing='ij')
    points=np.stack([i.ravel(),j.ravel()],axis=1)
    weights=points[:,0]*(nr-points[:,1])
    weights=weights/weights.sum()
    chosen=points[rng.choice(len(points),size=args.collocation,replace=False,p=weights)]
    xc=np.column_stack((t[chosen[:,0]],r[chosen[:,0],chosen[:,1]]))
    r0=np.logspace(-3,np.log10(R0),num=50)
    pinn=variant.PhysicsInformedModel(net,r0,t,r[:,-1],xc,M)
    print('WARNING: This trains the preserved experimental (alpha, m, rho0, v) variant,')
    print('not the final thesis (alpha, X, rho0, v) experiment.')
    pinn.train_adam(args.epochs)
    if args.lbfgs:
        pinn.train_lbfgs()
    if args.save_predictions:
        xt=np.column_stack((np.repeat(t,nr),r.ravel()))
        y=pinn.predict(xt)
        out=ROOT/'outputs'
        out.mkdir(exist_ok=True)
        target=out/'experimental_variant_predictions.npz'
        np.savez(target,U_pred=y)
        print(f'Predictions saved to {target} (columns alpha, m, rho0, v).')
    print('Finished. This is a model/train-loop sanity check, not a validated result.')

if __name__=='__main__':
    main()
