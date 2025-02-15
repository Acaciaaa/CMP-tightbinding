from __future__ import division
import kwant
from scipy.sparse.linalg import eigsh, eigs
from numpy.linalg import eigh
from scipy.linalg import kron
from scipy.optimize import minimize_scalar
# import pylab as py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys as system
import cmath
from math import sqrt, pi, sin, cos, dist
import warnings
from matplotlib.lines import Line2D

import KwantModel as km
def get_position_operator(sys, T, index):
    for i, site in enumerate(sys.sites):
        T[i, i] = site.pos[index]
    
class Localizer:
    def __init__(self, sys):
        H = sys.hamiltonian_submatrix(sparse=False)
        self.H = H
        self.dim = np.shape(H)[0]
        self.X = np.zeros(H.shape)
        get_position_operator(sys, self.X, 0)
        self.Y = np.zeros(H.shape)
        if km.model['name'] == km.HALDANE:
            get_position_operator(sys, self.Y, 1)

        self.H_part = kron(km.sz, self.H)
        self.X_part = kron(km.sx, self.X)
        self.Y_part = kron(km.sy, self.Y)
        
    def get_localizer(self, x=0, y=0, kappa=1):
        return self.H_part + kappa * (
            self.X_part + self.Y_part - kron(km.sx,x*np.identity(self.dim)) - kron(km.sy,y*np.identity(self.dim))
            )

def get_eigenvalues(sys, *args):
    x_list, y_list, kappa_list, num_eigvals = args[0],args[1],args[2], args[3]
    localizer = Localizer(sys)
    results = np.zeros((len(x_list),len(y_list),len(kappa_list),num_eigvals))
    for ix, x in enumerate(x_list):
        for iy, y in enumerate(y_list):
            for ikappa, kappa in enumerate(kappa_list):
                L = localizer.get_localizer(x=x, y=y, kappa=kappa)
                eigvals = eigsh(L, k=num_eigvals, sigma=0, return_eigenvectors=False, tol=1e-5)
                results[ix,iy,ikappa,:]=eigvals
    return results

def eigenvalues_change(name):
    if name == km.HALDANE:
        # h
        km.change_model(km.HALDANE, km.NOMASS)
        km.model['L']=km.model['W']=9
        h_list = np.array([1])
        # x y
        x_edge, y_edge = km.rectangle_vertex(km.model['L'], km.model['W'])
        num_cc = 50
        num_x = int((x_edge-x_edge+2)*num_cc)
        y_list=np.array([0])
        x_list=np.linspace(x_edge-2,x_edge,num=num_x)
        # kappa
        kappa_list = np.array([0.1,0.5,1])
        # num_eigvals
        num_eigvals = 20
    elif name == km.SSH:
        # h (t1)
        km.change_model(km.SSH, km.NONE)
        km.model['L']=9
        h_list = np.array([0.5])
        # x
        num_cc = 50
        num_x = int(3*num_cc)
        x_list=np.linspace(0,3,num=num_x)
        y_list=np.array([0])
        # kappa
        kappa_list = np.array([0.01,0.5,1,2,3])
        # num_eigvals
        num_eigvals = 10
    
    for ih, h in enumerate(h_list):
        km.model['h'] = h
        sys = km.model_builder()
        results = get_eigenvalues(sys,x_list,y_list,kappa_list,num_eigvals)
        plt.figure()
        plt.axhline(0, color='grey', linewidth=1, alpha=0.4)
        for line in range(num_eigvals):
            plt.scatter(x_list, results[:, 0, 4, line],s=1)
        plt.show()
        plt.close()

def edgestate_location_1d():
    # h
    km.change_model(km.SSH, km.NONE)
    km.model['L']=9
    h_list = np.array([0.5])
    # kappa
    num_kappa = 50
    kappa_list = np.linspace(0.5, 3, num_kappa)
    
    for ih, h in enumerate(h_list):
        km.model['h'] = h
        sys = km.model_builder()
        localizer = Localizer(sys)
        plt.figure()
        plt.grid(True, linestyle='--', alpha=0.4)
        location_change = np.zeros(num_kappa)
        
        def local_chern_number(x):
            L = localizer.get_localizer(x=x, y=0, kappa=kappa)
            eigvals = eigsh(L, k=1, sigma=0, return_eigenvectors=False, tol=1e-5)
            return eigvals[0]
        
        for ikappa, kappa in enumerate(kappa_list):
            result = minimize_scalar(lambda x: abs(local_chern_number(x)), bounds=(0, 0.5), method='bounded', options={'xatol': 1e-5})
            location_change[ikappa] = result.x
        plt.scatter(kappa_list, location_change, s=2, alpha=0.6)
        plt.ylabel('x')
        plt.ylim(0, 0.5)
        plt.xlabel('kappa')
        plt.title(f"ssh")
        plt.savefig(f"/Users/ruiqixu/Desktop/ssh_{km.model['L']}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
def edgestate_location_2d():
    # h
    km.change_model(km.HALDANE, km.NOMASS)
    km.model['L']=km.model['W']=9
    h_list = np.array([1])
    # x y -> 2 areas
    x_edge, y_edge = km.rectangle_vertex(km.model['L'], km.model['W'])
    x_list=np.array([0, 0.5, 1])
    y_list=np.array([0.5/sqrt(3), 1/sqrt(3), 2/sqrt(3)])
    # kappa
    num_kappa = 90
    kappa_list = np.linspace(0, 3, num_kappa)
    
    for ih, h in enumerate(h_list):
        km.model['h'] = h
        sys = km.model_builder()
        localizer = Localizer(sys)
        
        def binary_search(f, low, high, xtol):
            while (high - low) > xtol:
                mid = (low + high) / 2
                if f(mid) == 1 and f(high) == 0:
                    low = mid
                else:
                    high = mid
            return (low + high) / 2
        def local_chern_number(val, axis, kappa):
            if axis == 'x':
                L = localizer.get_localizer(x=val, y=fixed_value, kappa=kappa)
            else:  # axis == 'y'
                L = localizer.get_localizer(x=fixed_value, y=val, kappa=kappa)
            evals, _ = eigh(L)
            pos = np.sum(evals > 0)
            neg = np.sum(evals < 0)
            return ((pos - neg)/2)

        for axis, fixed_axis, fixed_list, edge_limit, edge_name in [('y', 'x', x_list, y_edge, 'upper'), ('x', 'y', y_list, x_edge, 'right')]:
            plt.figure()
            plt.grid(True, linestyle='--', alpha=0.4)
            for fixed_value in fixed_list:
                location_change = np.zeros(num_kappa)
                for ikappa, kappa in enumerate(kappa_list):
                    location_change[ikappa] = binary_search(lambda v: local_chern_number(v, axis, kappa), 0, edge_limit, xtol=1e-5)
                plt.scatter(kappa_list, location_change, label=f"{fixed_axis}={fixed_value:.2f}", s=2, alpha=0.6)
            plt.legend()
            plt.ylabel(axis)
            plt.ylim(0, edge_limit)
            plt.xlabel('kappa')
            plt.title(f"{edge_name} edge")
            plt.savefig(f"/Users/ruiqixu/Desktop/{edge_name}_{km.model['L']}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
#eigenvalues_change(km.SSH)
#edgestate_location_2d()
edgestate_location_1d()