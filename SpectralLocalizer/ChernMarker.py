from __future__ import division
import kwant
from scipy.sparse.linalg import eigsh, eigs
from numpy.linalg import eigh
from scipy.linalg import kron
from scipy.optimize import minimize_scalar
from scipy.spatial import KDTree
# import pylab as py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys as system
import cmath
from math import sqrt, pi, sin, cos, dist, isclose
import warnings
from matplotlib.lines import Line2D

import KwantModel as km

class ChernMarker:
    # fix position info
    def __init__(self, L, W):
        #km.change_model(km.HALDANE, km.NOMASS)
        km.change_model(km.DEFECT, km.SINGLE)
        self.L=km.model['L']=L
        self.W=km.model['W']=W
        self.cc = km.model['cc']
        self.ll = self.cc/sqrt(3)
        self.tmp_sys=km.model_builder()
        self.dim = len(self.tmp_sys.sites)
        self.X, self.Y = np.zeros((self.dim, self.dim)),np.zeros((self.dim, self.dim))
        for i, site in enumerate(self.tmp_sys.sites):
            self.X[i, i] = site.pos[0]
            self.Y[i, i] = site.pos[1]
    
    def path_1d(self):
        # direction: y: zigzag->unitcell path; x: armchair->pure A/B path
        # [y, mid(longer)][y, dev(shorter)][x, a][x, b]
        self.all_paths = [('y', 'mid'),('y', 'dev'), ('x', 'b'),('x', 'a')]
        self.path_number = len(self.all_paths)
        self.path_pos = []
        tol = 1e-6
        (x, y) = km.rectangle_vertex(self.L, self.W)
        for (direction, length) in self.all_paths:
            tmp_path_pos = []
            if direction == 'x' and length == 'a':
                for i in range(self.L+1):
                    tmp_path_pos.append([-x+i*self.cc, -self.ll/2])
                assert isclose(tmp_path_pos[-1][0],x,abs_tol=tol), "(x, a) path_pos ERROR!"
            elif direction == 'x' and length == 'b':
                for i in range(self.L+1):
                    tmp_path_pos.append([-x+i*self.cc, self.ll/2])
                assert isclose(tmp_path_pos[-1][0],x,abs_tol=tol), "(x, b) path_pos ERROR!"
            elif direction == 'y' and length == 'mid':
                # preliminary: mid is longer, dev is shorter: (L-1)%4=0
                for i in range(self.W+1):
                    tmp_path_pos.append([0, -y+((i+1)//2*3-i%2)*self.ll])
                assert isclose(tmp_path_pos[-1][1],y,abs_tol=tol), "(y, mid) path_pos ERROR!"
            elif direction == 'y' and length == 'dev':
                # preliminary: mid is longer, dev is shorter: (L-1)%4=0
                for i in range(self.W+1):
                    tmp_path_pos.append([self.cc/2, -y+self.ll/2+((i+1)//2*3-i%2*2)*self.ll])
                assert isclose(tmp_path_pos[-1][1],y-self.ll/2,abs_tol=tol), "(y, dev) path_pos ERROR!"
            self.path_pos.append(tmp_path_pos)
        
        self.path_index = [np.zeros(len(self.path_pos[ipath]), dtype=int) for ipath in range(self.path_number)]
        for isite, site in enumerate(self.tmp_sys.sites):
            for ipath in range(self.path_number):
                tree = KDTree(self.path_pos[ipath])
                dist, index = tree.query([site.pos[0], site.pos[1]])
                if dist < tol:
                    self.path_index[ipath][index] = isite
                    
    def chern_operator(self, sys):
        H = sys.hamiltonian_submatrix(sparse=False)
        energies, eigenstates = eigh(H)
        occupied_states = eigenstates[:, energies < 0]
        P = occupied_states @ occupied_states.conj().T
        commut_X_P = self.X @ P - P @ self.X
        commut_Y_P = self.Y @ P - P @ self.Y
        chern_operator = np.imag(P @ commut_X_P @ commut_Y_P)
        area_uc = np.sqrt(3) / 2 / 2
        return chern_operator * -4 * np.pi / area_uc
    
    def draw_chern_marker_1d(self, sys, macroscopic_average=None):
        if macroscopic_average is not None:
            pass
        C = self.chern_operator(sys)
        for ipath in range(self.path_number):
            plt.figure()
            plt.axhline(0, color='grey', linestyle='--',alpha=0.5)
            plt.axhline(1, color='grey', linestyle='--',alpha=0.5)
            plt.axhline(-1, color='grey', linestyle='--',alpha=0.5)
            chern_list = np.array([C[self.path_index[ipath][i], self.path_index[ipath][i]] for i in range(len(self.path_index[ipath]))])
            if self.all_paths[ipath][0] == 'y':
                pos_list = np.array(self.path_pos[ipath])[:, 1]
                x_min, x_max = np.floor(min(pos_list)), np.ceil(max(pos_list))
                plt.xlabel('y')
                plt.xticks(np.arange(x_min, x_max + 2*self.ll, 2*self.ll), fontsize=8)
                plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                plt.title(f"{km.model['name']} h{km.model['h']} L{self.L} W{self.W} zigzag->{self.all_paths[ipath][1]}")
            elif self.all_paths[ipath][0] == 'x':
                pos_list = np.array(self.path_pos[ipath])[:, 0]
                x_min, x_max = np.floor(min(pos_list)), np.ceil(max(pos_list))
                plt.xlabel('x')
                plt.xticks(np.arange(x_min, x_max + self.cc, self.cc), fontsize=8)
                plt.title(f"{km.model['name']} h{km.model['h']} L{self.L} W{self.W} armchair->{self.all_paths[ipath][1]}")
            
            plt.scatter(pos_list, chern_list)
            y_min, y_max = np.floor(min(chern_list)), np.ceil(max(chern_list))
            plt.yticks(np.arange(y_min, y_max + 1, 1))
            plt.ylabel('chern marker')
            #plt.show()
            plt.savefig(f"/Users/ruiqixu/Desktop/kappa/chern marker/{km.model['name']}/1d/{km.model['h']}{self.all_paths[ipath][1]}.png", dpi=300, bbox_inches='tight')
            
    def draw_chern_marker_2d(self, sys, macroscopic_average=None):
        if macroscopic_average is not None:
            pass
        C = self.chern_operator(sys)
        x_list, y_list, chern_list = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        for isite, site in enumerate(sys.sites):
            x_list[isite] = site.pos[0]
            y_list[isite] = site.pos[1]
            chern_list[isite] = C[isite, isite]
        plt.figure()
        sc = plt.scatter(x_list, y_list, c=chern_list, cmap='coolwarm', vmin=-2, vmax=2, edgecolors='none')
        plt.colorbar(sc, label="C#")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{km.model['name']} h{km.model['h']} L{self.L} W{self.W}")
        plt.show()
        #plt.savefig(f"/Users/ruiqixu/Desktop/kappa/chern marker/{km.model['name']}/2d/{km.model['h']}.png", dpi=300, bbox_inches='tight')
    
L = 25
W = 25
marker = ChernMarker(L, W)
#marker.path_1d()
#km.change_model(km.HALDANE, km.NOMASS)
km.change_model(km.DEFECT, km.SINGLE)
km.model['L']=L
km.model['W']=W
km.model['h']=1.1
sys=km.model_builder()
#marker.draw_chern_marker_1d(sys)
marker.draw_chern_marker_2d(sys)