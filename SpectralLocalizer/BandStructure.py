#!/usr/bin/env python

# Haldane model from Phys. Rev. Lett. 61, 2015 (1988)
# Calculates density of states for finite sample of Haldane model

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

def model_build(m, t1, t2, t3, tc, m2):
    lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
    orb=[[0, 0],[-1/3, 2/3], [1/3, 1/3]]

    my_model=tb_model(2,2,lat,orb)

    my_model.set_onsite([m, -m, m2])
    # (amplitude, i, j, [lattice vector to cell containing j])
    neighbors_graphene = [[0, 0], [0, 1], [-1, 1]]
    for neighbor in neighbors_graphene:
        my_model.set_hop(-t1, 1, 0, neighbor)
            
    neighbors_triangular = [[0, 1], [-1, 0], [-1, 1]]
    for neighbor in neighbors_triangular:
        my_model.set_hop(-t2, 2, 2, neighbor)
            
    neighbors_second = [[0, 0], [1, 0], [0, 1]]
    for neighbor in neighbors_second:
        my_model.set_hop(-t3, 2, 0, neighbor)
    neighbors_second = [[0, 0], [1, 0], [1, -1]]
    for neighbor in neighbors_second:
        my_model.set_hop(-t3, 2, 1, neighbor)
        
    temp = tc*np.exp((1.j)*np.pi/2.)
    neighbors_imag = [[0, 1], [-1, 0], [1, -1]]
    for neighbor in neighbors_imag:
        my_model.set_hop(temp.conjugate(), 1, 1, neighbor)
        my_model.set_hop(temp, 0, 0, neighbor)
        
    return my_model
        
def band_periodic(**kwargs):
    my_model = model_build(**kwargs)
    path=[[0.,0.],[2./3.,1./3.],[.5,.5],[1./3.,2./3.], [0.,0.]]
    label=(r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $')
    (k_vec,k_dist,k_node)=my_model.k_path(path,101)
    evals=my_model.solve_all(k_vec)

    plt.figure()
    fig, ax = plt.subplots()
    ax.set_xlim(k_node[0],k_node[-1])
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    for n in range(len(k_node)):
        ax.axvline(x=k_node[n],linewidth=0.5, color='k')
    ax.set_title("periodic band structure")
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel("Band energy")
    for n in range(3):
        ax.plot(k_dist,evals[n])

    p_text = ', '.join([f'{key}: {value:.3f}' if isinstance(value, float) else f'{key}: {value}' for key, value in kwargs.items()])
    plt.figtext(0.5, 0.01, p_text, ha="center", va="bottom", fontsize=10, color="blue", bbox=dict(facecolor='lightblue', edgecolor='blue'))
    filename = f"t3={kwargs['t3']:.2f}_tc={kwargs['tc']:.2f}_t2={kwargs['t2']:.2f}_m2={kwargs['m2']:.2f}"
    plt.savefig(f'./SpectralLocalizer/band/ribbon_{filename}.png')
    
def band_ribbon(**kwargs):
    tmp_model = model_build(**kwargs)
    my_model=tmp_model.cut_piece(10,0,glue_edgs=False)
    
    (k_vec,k_dist,k_node)=my_model.k_path([[0.0], [1.0]],200,report=False)
    
    plt.figure()
    fig, ax = plt.subplots()
    # solve model on all of these k-points
    eval = my_model.solve_all(k_vec,eig_vectors=False)
    for band in range(eval.shape[0]):
        ax.plot(k_dist,eval[band,:],"k-",linewidth=0.5)
    ax.set_title("ribbon band structure")
    ax.set_xlabel("Path in k-vector")
    ax.set_ylabel("Band energies")
    
    p_text = ', '.join([f'{key}: {value:.3f}' if isinstance(value, float) else f'{key}: {value}' for key, value in kwargs.items()])
    plt.figtext(0.5, 0.01, p_text, ha="center", va="bottom", fontsize=10, color="blue", bbox=dict(facecolor='lightblue', edgecolor='blue'))
    filename = f"t3={kwargs['t3']:.2f}_tc={kwargs['tc']:.2f}_t2={kwargs['t2']:.2f}_m2={kwargs['m2']:.2f}"
    plt.savefig(f'./SpectralLocalizer/band/periodic_{filename}.png')
    
for (t3, tc, t2, m2) in [(0.1, 0.1, 0.8, -0.1),
                         (0.1, 0.5, 1, -0.5),
                         (0.5, 0.1, 0.5, -0.1),
                         (1, 0.1, 0.1, -0.1),
                         (1, 0.1, 1, -0.5),
                         (1, 1, 1, -0.8)]:
        band_periodic(m=0.0, t1=1.0, t2=t2, t3=t3, tc=tc, m2=m2)
        band_ribbon(m=0.0, t1=1.0, t2=t2, t3=t3, tc=tc, m2=m2)