#!/usr/bin/env python3
"""Minimal forward/backprop check of the surviving experimental PyTorch model.

This is NOT a reproduction of the final thesis training or accuracy.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch

p=Path(__file__).resolve().parents[1]/'research_code/PINN_OS_4.py'
s=importlib.util.spec_from_file_location('pinn_variant',p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
net=m.NN_arch([2,8,8,4],lb=np.array([0.,0.]),ub=np.array([50.,10.]),M=1.,R0=10.)
net.train()
device=next(net.parameters()).device
x=torch.tensor([[0.,1.],[10.,2.],[20.,3.]],dtype=torch.float32,device=device)
y=net(x)
assert y.shape==(3,4) and torch.isfinite(y).all()
assert ((y[:,0]>0)&(y[:,0]<1)).all()
assert ((y[:,1]>0)&(y[:,1]<1)).all()
assert (y[:,2]>0).all()
assert ((y[:,3]<0)&(y[:,3]>-1)).all()
y.square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
print(f'Archived model forward/backward smoke test PASSED on {device}')
print('This tests PyTorch computation only, not final thesis reproducibility.')
