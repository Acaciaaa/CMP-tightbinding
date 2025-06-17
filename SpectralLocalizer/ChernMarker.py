from __future__ import division
import kwant
from scipy.sparse.linalg import eigsh, eigs
from numpy.linalg import eigh
from scipy.linalg import kron
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from scipy.signal import argrelextrema
from scipy.constants import e, hbar, c
# import pylab as py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import matplotlib.colors as mcolors
import numpy as np
from tabulate import tabulate
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys as system
import cmath
import h5py
from math import sqrt, pi, sin, cos, dist, isclose
import warnings
from matplotlib.lines import Line2D

import KwantModel as km

class Cstorage:
    def __init__(self, L):
        self.L=L
        self.file_path=f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/C/{L}/'
        C_file='C_diag.h5'
        self.C_file = self.file_path + C_file
        self.num_h = 400
            
    def write_C(self):
        marker = ChernMarker(self.L, self.L)
        km.change_model(km.DEFECT, km.SINGLE)
        km.model['L']=self.L
        km.model['W']=self.L
        h_list = np.linspace(0, 2, self.num_h)
        with h5py.File(self.C_file, 'a') as C_file:
            if 'C' not in C_file:
                C_diag = C_file.create_dataset('C', (self.num_h, marker.dim), dtype='float64')
            else:
                C_diag = C_file['C']
            
            written_rows = np.any(C_diag[...] != 0, axis=1)
            start_ih = np.sum(written_rows)
            
            for ih in range(start_ih, self.num_h):
                km.model['h'] = h_list[ih]
                sys = km.model_builder()
                C = marker.crosshair_operator(sys)
                C_diag[ih, ...] = np.diag(C)

                C_file.flush()

    
    def read_C_unit(self, h_index):
        with h5py.File(self.C_file, 'r') as C_file:
            C_diag = C_file['C'][h_index,...]
        return C_diag

class ChernMarker:
    # fix position info
    def __init__(self, L, W):
        #km.change_model(km.HALDANE, km.NOMASS)
        km.change_model(km.DEFECT, km.SINGLE)
        self.L=km.model['L']=L
        self.W=km.model['W']=W
        self.cc = km.model['cc']
        self.ll = self.cc/sqrt(3)
        self.area_uc = np.sqrt(3) / 2 / 2
        self.area_sample = np.sqrt(3) / 2 * (L*W-(W-1)/2)
        self.tmp_sys=km.model_builder()
        self.dim = len(self.tmp_sys.sites)
        self.X, self.Y = np.zeros((self.dim, self.dim)),np.zeros((self.dim, self.dim))
        for i, site in enumerate(self.tmp_sys.sites):
            self.X[i, i] = site.pos[0]
            self.Y[i, i] = site.pos[1]
    
    def path_1d(self):
        # direction: y: zigzag->unitcell path; x: armchair->pure A/B path
        # [y, mid(longer)][y, dev(shorter)][x, a][x, b]
        self.all_paths = [('y', 'mid'),('x', 'a')]
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
                    
    def occupied_projector(self, sys):
        self.H = sys.hamiltonian_submatrix(sparse=False)
        energies, eigenstates = eigh(self.H)
        occupied_states = eigenstates[:, energies < 0]
        P = occupied_states @ occupied_states.conj().T
        return P
    def crosshair_operator(self, sys, allow_range = None):
        P = self.occupied_projector(sys)
        self.Xb, self.Yb = (
            np.clip(self.X, allow_range[0][0], allow_range[0][1]),
            np.clip(self.Y, allow_range[1][0], allow_range[1][1])
        ) if allow_range is not None else (self.X, self.Y)
        crosshair_operator = -np.imag(P@self.Xb@P@self.Yb@P)
        return crosshair_operator * 4 * np.pi / self.area_uc
    def get_allow_range(self, edge_x=0, edge_y=0):
        epsilon = 0.01
        (x, y) = km.rectangle_vertex(self.L, self.W)
        if edge_x == 0 and edge_y == 0:
            return [[-x-epsilon, x+epsilon],[-y-epsilon, y+epsilon]]
        cutoff_x = edge_x*self.cc/2-self.cc/4
        cutoff_y = (edge_y-1)//2 * 1.5*self.ll + (edge_y-1)%2 * self.ll/2 + self.ll/4
        allow_range=[[-x+cutoff_x, x-cutoff_x], [-y+cutoff_y, y-cutoff_y]]
        return allow_range
    
    def test_time(self):
        pass
        # km.model['h'] = 0.5
        # sys = km.model_builder()
        # P = self.occupied_projector(sys)
        # lp = LineProfiler()
        # lp.add_function(self.crosshair_operator)
        # lp.enable_by_count()
        # C = self.crosshair_operator(sys, allow_range=None, provide_P=P)
        # lp.print_stats()
    
    def draw_total(self):
        L=25
        pstorage = Cstorage(L)
        h_list = np.linspace(0, 2, pstorage.num_h)
        total_list, star_list, nostar_list = np.zeros(pstorage.num_h), np.zeros(pstorage.num_h), np.zeros(pstorage.num_h)
        X_diag = np.diag(self.X)
        Y_diag = np.diag(self.Y)
        mask_star = np.sqrt(X_diag**2 + Y_diag**2) < (self.ll+0.1)
        def get_mask(edge):
            bulk_range = self.get_allow_range(int(self.L*edge), int(self.W*edge))
            mask_total = (bulk_range[0][0] < X_diag) & (X_diag < bulk_range[0][1]) & (bulk_range[1][0] < Y_diag) & (Y_diag < bulk_range[1][1])
            return mask_total
        
        
        def get_valley_hc():
            edge_list = [0.2, 0.3, 0.4, 0.5, 'star']
            valley_list, hc_list = np.zeros(5), np.zeros(5)
            for iedge, edge in enumerate(edge_list):
                tmp_list = np.zeros(num_h)
                mask = mask_star if edge == 'star' else get_mask(edge)
                for ih in range(num_h):
                    C_diag = pstorage.read_C_unit(ih)
                    tmp_list[ih] = np.mean(C_diag[mask])
                # remember we want h
                minima_indices = argrelextrema(tmp_list, np.less)[0]
                valley_list[iedge] = h_list[minima_indices[0]] if minima_indices.size > 0 else None
                zero_crossing_indices = np.where((tmp_list[:-1] > 0) & (tmp_list[1:] < 0))[0]
                hc_list[iedge] = (h_list[zero_crossing_indices[0]]+h_list[zero_crossing_indices[0]+1])/2 if zero_crossing_indices.size > 0 else None
            data = [valley_list, hc_list]
            headers = edge_list
            row_indices = ["valley", "hc"]
            print(tabulate(data, headers=headers, tablefmt="grid", showindex=row_indices, floatfmt=".3f"))
        def diff_edge(way):
            plt.figure()
            ax = plt.gca()
            plt.axhline(0, color='grey', linestyle='--',alpha=0.5)
            for edge in [0.2, 0.3, 0.4, 0.5]:
                mask_total = get_mask(edge)
                for ih in range(num_h):
                    C_diag = pstorage.read_C_unit(ih)
                    total_list[ih] = way(C_diag[mask_total])
                plt.plot(h_list, total_list, linestyle='-', label = f"{edge:.2f}")
            plt.legend()
            plt.xlabel('h')
            if way is np.sum:
                way_label = 'sum'
                plt.ylim(-300, 300)
            elif way is np.mean:
                way_label = 'avg'
                plt.ylim(-0.45, 0.45)
            plt.ylabel(way_label+' M1')        
            plt.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in')
            plt.xticks(np.arange(0, 2.1, 0.1))
            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            plt.show()
            # plt.savefig(
            #     f"/Users/ruiqixu/Desktop/kappa/chern marker/{km.model['name']}/total/M2011/edge_{way_label}.png", 
            #     dpi=300, bbox_inches='tight')
        def diff_region(way):
            fixed_edge, fixed_color = 0.2, 'dodgerblue'
            plt.figure()
            ax = plt.gca()
            plt.axhline(0, color='grey', linestyle='--',alpha=0.5)
            mask_total = get_mask(fixed_edge)
            for ih in range(num_h):
                C_diag = pstorage.read_C_unit(ih)
                total_list[ih] = way(C_diag[mask_total])
                star_list[ih] = way(C_diag[mask_star])
                nostar_list[ih] = way(C_diag[mask_total&~mask_star])
            plt.plot(h_list, total_list, linestyle='-',color=fixed_color, label = "total")
            plt.plot(h_list, star_list, linestyle=':',color=fixed_color, label = "star")
            plt.plot(h_list, nostar_list, linestyle='--',color=fixed_color, label = "without star")
            plt.legend()
            if way is np.sum:
                way_label = 'sum'
                plt.ylim(-300, 300)
            elif way is np.mean:
                way_label = 'avg'
            plt.ylabel(way_label+' M1')
            plt.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in')
            plt.xticks(np.arange(0, 2.1, 0.1))
            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            plt.show()
            # plt.savefig(
            #     f"/Users/ruiqixu/Desktop/kappa/chern marker/{km.model['name']}/total/M2011/region_{way_label}.png", 
            #     dpi=300, bbox_inches='tight')
        # diff_edge(np.sum)
        # diff_edge(np.mean)
        # diff_region(np.sum)
        # diff_region(np.mean)
        # get_valley_hc()

    def draw_marker_1d(self, sys):
        edges = [0, 1/8, 1/4, 1/2]
        for edge in edges:
            allow_range = self.get_allow_range(int(self.L*edge), int(self.W*edge))
            C = self.crosshair_operator(sys, allow_range)
            for ipath in range(self.path_number):
                plt.figure()
                ax = plt.gca()
                plt.axhline(0, color='grey', linestyle='--',alpha=0.5)
                plt.axhline(1, color='grey', linestyle='--',alpha=0.5)
                plt.axhline(-1, color='grey', linestyle='--',alpha=0.5)
                plt.axvline(0, color='pink', linestyle='--', alpha=0.5)
                chern_list = np.array([C[self.path_index[ipath][i], self.path_index[ipath][i]] for i in range(len(self.path_index[ipath]))])
                if self.all_paths[ipath][0] == 'y':
                    # only for ('y', 'mid')
                    plt.axvline(allow_range[1][0], color='crimson', linestyle='--', alpha=0.5)
                    plt.axvline(allow_range[1][1], color='crimson', linestyle='--', alpha=0.5)
                    pos_list = np.array(self.path_pos[ipath])[:, 1]
                    plt.xlabel(f"y (W={self.W})")
                    pairs = pos_list.reshape(-1, 2)
                    q1 = pairs[:, 0] + (pairs[:, 1] - pairs[:, 0]) / 4
                    q2 = pairs[:, 1] - (pairs[:, 1] - pairs[:, 0]) / 4
                    inserted_ticks = np.hstack((q1, q2))
                    inserted_ticks.sort()
                    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                    plt.text(0.97, 0.97, f"h={km.model['h']} zigzag", fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)
                elif self.all_paths[ipath][0] == 'x':
                    plt.axvline(allow_range[0][0], color='crimson', linestyle='--', alpha=0.5)
                    plt.axvline(allow_range[0][1], color='crimson', linestyle='--', alpha=0.5)
                    pos_list = np.array(self.path_pos[ipath])[:, 0]
                    plt.xlabel(f"x (L={self.L})")
                    inserted_ticks = (pos_list[:-1] + pos_list[1:]) / 2
                    plt.text(0.97, 0.97, f"h={km.model['h']} armchair", fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)
                
                plt.xticks(pos_list)
                ax.set_xticks(inserted_ticks, minor=True)
                ax.set_xticklabels([])
                ax.tick_params(axis='both', which='both', direction='in')
                plt.scatter(pos_list, chern_list)
                y_min, y_max = np.floor(min(chern_list)), np.ceil(max(chern_list))
                plt.yticks(np.arange(y_min, y_max + 1, 1))
                plt.ylim(-3, 3)
                plt.ylabel(f"M1")
                plt.show()
                #plt.savefig(f"/Users/ruiqixu/Desktop/kappa/chern marker/{km.model['name']}/1d/{km.model['h']}{self.all_paths[ipath][1]}_{edge:.2f}.png", dpi=300, bbox_inches='tight')
                
    def draw_marker_2d(self, sys):
        edges = [0, 1/8, 1/4, 1/2]
        for edge in edges:
            allow_range = self.get_allow_range(int(self.L*edge), int(self.W*edge))
            C = self.crosshair_operator(sys, allow_range)
            x_list, y_list, chern_list = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
            for isite, site in enumerate(sys.sites):
                x_list[isite] = site.pos[0]
                y_list[isite] = site.pos[1]
                chern_list[isite] = C[isite, isite]
            plt.figure()
            ax = plt.gca()
            ax.add_patch(patches.Rectangle(
                (allow_range[0][0], allow_range[1][0]), allow_range[0][1]-allow_range[0][0], allow_range[1][1] - allow_range[1][0], 
                fill=False, edgecolor='green', alpha=0.8, linewidth=2,
                ))
            sc = plt.scatter(x_list, y_list, c=chern_list, cmap='coolwarm', vmin=-3, vmax=3, edgecolors='none', s=50)
            
            x_max = np.ceil(max(x_list))
            y_max = np.ceil(max(y_list))
            plt.xlim(0, x_max+0.1)
            plt.ylim(0, y_max+0.5)
            plt.colorbar(sc, label="C#")
            plt.axis('off')
            plt.title(f"{km.model['name']} h={km.model['h']} L={self.L} W={self.W}")
            plt.show()
            #plt.savefig(f"/Users/ruiqixu/Desktop/kappa/chern marker/{km.model['name']}/2d/{km.model['h']}_{edge:.2f}.png", dpi=300, bbox_inches='tight')
        
    def orbital_magnetization(self, sys):
        P = self.occupied_projector(sys)
        M_old = np.imag(P@self.X@self.H@self.Y@P)
        Q = np.eye(self.dim) - P
        M_local = np.imag(P@self.X@Q@self.H@Q@self.Y@P)
        M_itinerant = -np.imag(Q@self.X@P@self.H@P@self.Y@Q)
        M_new = M_local+M_itinerant
        return np.diag(M_old)/self.area_sample, np.diag(M_local)/self.area_sample, np.diag(M_itinerant)/self.area_sample
    
    def magnetization_phi(self):
        num_phi = 200
        phi_list = np.linspace(0, 1, num_phi)
        M_old_trace, M_new_trace = np.zeros(num_phi), np.zeros(num_phi)
        plt.figure()
        ax = plt.gca()
        plt.axhline(0, color='grey', linestyle='--',alpha=0.5)
        for iphi, phi in enumerate(phi_list):
            km.model['phi'] = phi
            sys = km.model_builder()
            M_old, M_local, M_itinerant = self.orbital_magnetization(sys)
            M_old_trace[iphi], M_new_trace[iphi] = np.sum(M_old), np.sum(M_local+M_itinerant)
        plt.scatter(phi_list, M_old_trace, label='M old', color='red', alpha=0.8)
        plt.scatter(phi_list, M_new_trace, label='M new', color='blue', alpha=0.8)
        plt.legend()
        plt.xlabel(r'$\phi(/\pi)$')
        plt.ylabel('orbital magnetization')
        plt.savefig(f"/Users/ruiqixu/Desktop/kappa/chern marker/nontopology/{km.model['name']}/diffphi_{num_phi}_{km.model['h']}.png", dpi=300, bbox_inches='tight')
        #plt.show()
            
    def draw_magnetization_1d(self, sys):
        M_old, M_local, M_itinerant = self.orbital_magnetization(sys)
        for ipath in range(self.path_number):
            plt.figure()
            ax = plt.gca()
            plt.axhline(0, color='grey', linestyle='--',alpha=0.5)
            M_old_list = np.array([M_old[self.path_index[ipath][i]] for i in range(len(self.path_index[ipath]))])
            M_local_list = np.array([M_local[self.path_index[ipath][i]] for i in range(len(self.path_index[ipath]))])
            M_itinerant_list = np.array([M_itinerant[self.path_index[ipath][i]] for i in range(len(self.path_index[ipath]))])
            if self.all_paths[ipath][0] == 'y':
                # only for ('y', 'mid')
                pos_list = np.array(self.path_pos[ipath])[:, 1]
                plt.xlabel(f"y (W={self.W}) h={km.model['h']} zigzag")
                pairs = pos_list.reshape(-1, 2)
                q1 = pairs[:, 0] + (pairs[:, 1] - pairs[:, 0]) / 4
                q2 = pairs[:, 1] - (pairs[:, 1] - pairs[:, 0]) / 4
                inserted_ticks = np.hstack((q1, q2))
                inserted_ticks.sort()
                #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            elif self.all_paths[ipath][0] == 'x':
                pos_list = np.array(self.path_pos[ipath])[:, 0]
                plt.xlabel(f"x (L={self.L}) h={km.model['h']} armchair")
                inserted_ticks = (pos_list[:-1] + pos_list[1:]) / 2
                
            plt.xticks(pos_list)
            ax.set_xticks(inserted_ticks, minor=True)
            ax.set_xticklabels([])
            ax.tick_params(axis='both', which='both', direction='in')
            
            plt.text(0.5, 1.03, f"Tr(M_old): {np.sum(M_old):.3e}, Tr(M_new): {np.sum(M_local+M_itinerant):.3e}", ha='center', va='center', transform=plt.gca().transAxes)
            plt.scatter(pos_list, M_old_list, label = 'M old', color='red', alpha=0.8)
            plt.scatter(pos_list, M_local_list + M_itinerant_list, label = 'M new', color='blue', alpha=0.8)
            
            # plt.scatter(pos_list, M_local_list, label = 'M local', color='orange', alpha=0.8)
            # plt.scatter(pos_list, M_itinerant_list, label = 'M itinerant', color='purple', alpha=0.8)
            
            plt.legend()
            plt.ylabel("orbital magnetization")
            #plt.show()
            plt.savefig(f"/Users/ruiqixu/Desktop/kappa/chern marker/nontopology/{km.model['name']}/old_new_{km.model['h']}{self.all_paths[ipath][1]}.png", dpi=300, bbox_inches='tight')
            #plt.savefig(f"/Users/ruiqixu/Desktop/kappa/chern marker/nontopology/{km.model['name']}/local_itinerant_{km.model['h']}{self.all_paths[ipath][1]}.png", dpi=300, bbox_inches='tight')
                
    
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

#L = 25
#W = 25
#marker = ChernMarker(L, W)
#marker.path_1d()
#km.change_model(km.HALDANE, km.NOMASS)
#km.change_model(km.DEFECT, km.SINGLE)
#km.model['L']=L
#km.model['W']=W
#km.model['h']=0.5
#sys=km.model_builder()
#marker.draw_marker_1d(sys)
#marker.draw_marker_2d(sys)
#marker.draw_total()
#marker.draw_magnetization_1d(sys)
#marker.magnetization_phi()

# pstorage = Cstorage(37)
# pstorage.write_C()

def store_hc():
    h_list, h_29_list = np.linspace(0, 2, 400),np.linspace(0, 2, 200)
    hc_list = np.zeros((16, 4))
    a_list = [0.5, 1, 2, 4]
    L_list = [9, 13, 17, 21, 25, 29, 33, 37]
    edge_list = [0.2, 0.3, 0.4, 0.5]
    for iL, L in enumerate(L_list):
        big_flow_list = np.load(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/big_flow_list_correction.npy')
        for ia, a in enumerate(a_list):
            crossing_index = np.where(big_flow_list[3, ia] > 0)[0][-1]
            if L >= 29:
                zero=(h_29_list[crossing_index]+h_29_list[crossing_index+1])/2
            else:
                zero=(h_list[crossing_index]+h_list[crossing_index+1])/2
            hc_list[iL, ia] = zero
    for iL, L in enumerate(L_list):
        pstorage = Cstorage(L)
        marker = ChernMarker(L, L)
        X_diag = np.diag(marker.X)
        Y_diag = np.diag(marker.Y)
        for iedge, edge in enumerate(edge_list):
            bulk_range = marker.get_allow_range(int(L*edge), int(L*edge))
            mask = (bulk_range[0][0] < X_diag) & (X_diag < bulk_range[0][1]) & (bulk_range[1][0] < Y_diag) & (Y_diag < bulk_range[1][1])
            tmp_list = np.zeros(pstorage.num_h)
            for ih in range(pstorage.num_h):
                C_diag = pstorage.read_C_unit(ih)
                tmp_list[ih] = np.sum(C_diag[mask])
            crossing_index = np.where(tmp_list > 0)[0][-1]
            zero=(h_list[crossing_index]+h_list[crossing_index+1])/2
            hc_list[8+iL, iedge] = zero
    np.save(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/hc_list.npy', hc_list)

def hc(ax):
    a_list = [0.5, 1, 2.0, 4]
    L_list = [9, 13, 17, 21, 25, 29, 33, 37]
    edge_list = [0.2, 0.3, 0.4, 0.5]
    hc_list = np.load('/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/hc_list.npy')
    for iedge, edge in enumerate(edge_list):
        if edge == 0.2:
            continue
        ax.plot(L_list, hc_list[8:, iedge], marker='o', linestyle='--', color='skyblue',markersize=3,linewidth=1,alpha=0.7)
    for iedge, edge in enumerate(edge_list):
        if edge == 0.2:
            ax.plot(L_list, hc_list[8:, iedge], label = rf'edge$={edge}$', marker='o', linestyle='-', color='mediumblue',markersize=3.5,linewidth=2,alpha=1)
            break
    for ia, a in enumerate(a_list):
        if a == 2:
            continue
        ax.plot(L_list, hc_list[:8, ia], marker='o', linestyle='--', color='pink',markersize=3,linewidth=1,alpha=0.7)
    for ia, a in enumerate(a_list):
        if a == 2:
            ax.plot(L_list, hc_list[:8, ia], label = rf'$\alpha={a}$', marker='o', linestyle='-', color='crimson',markersize=3.5,linewidth=2,alpha=1)
            break        
    ax.tick_params(direction='in', which='both')
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$h_c$")
    ax.set_xticks(ticks=L_list)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.5)
    ax.legend(frameon=False)
    
def zoomin(ax):
    h_list = np.linspace(0, 2, 400)
    region =  0.2
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)][::-1]
    
    ax.axhline(0, color='grey', linewidth=1.5, linestyle='--',alpha=0.6)
    for iL, L in enumerate([9, 13, 17, 21, 25, 29, 33, 37]):
        pstorage = Cstorage(L)
        marker = ChernMarker(L, L)
        X_diag = np.diag(marker.X)
        Y_diag = np.diag(marker.Y)
        bulk_range = marker.get_allow_range(int(L*region), int(L*region))
        mask = (bulk_range[0][0] < X_diag) & (X_diag < bulk_range[0][1]) & (bulk_range[1][0] < Y_diag) & (Y_diag < bulk_range[1][1])
       
        tmp_list = np.zeros(pstorage.num_h)
        for ih in range(pstorage.num_h):
            C_diag = pstorage.read_C_unit(ih)
            tmp_list[ih] = np.sum(C_diag[mask])
        ax.plot(h_list, tmp_list, label=rf'$L={L}$',color=colors[iL], linewidth=2, alpha = 0.8)
    
    ax.set_xticks(np.arange(0, 2.1, 0.1))
    ax.set_yticks(ticks=[-40, 0, 40])
    ax.tick_params(direction='in', which='both')
    # ax.set_xlabel(r"$h$")
    # ax.set_ylabel(r"$M$")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylim(-40, 40)
    ax.set_xlim(0.89, 1.11)

from GenerateCSV import generate_csv
def marker_diffsize(ifcsv=False):
    h_list = np.linspace(0, 2, 400)
    region =  0.2
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)][::-1]
    data_sets = []
    
    plt.figure()
    plt.axhline(0, color='grey', linewidth=1, linestyle='--',alpha=0.5)
    for iL, L in enumerate([9, 13, 17, 21, 25, 29, 33, 37]):
        pstorage = Cstorage(L)
        marker = ChernMarker(L, L)
        X_diag = np.diag(marker.X)
        Y_diag = np.diag(marker.Y)
        bulk_range = marker.get_allow_range(int(L*region), int(L*region))
        mask = (bulk_range[0][0] < X_diag) & (X_diag < bulk_range[0][1]) & (bulk_range[1][0] < Y_diag) & (Y_diag < bulk_range[1][1])
        region_text = rf'edge$={region}$'
            
        tmp_list = np.zeros(pstorage.num_h)
        for ih in range(pstorage.num_h):
            C_diag = pstorage.read_C_unit(ih)
            tmp_list[ih] = np.sum(C_diag[mask])
        plt.plot(h_list, tmp_list, label=rf'$L={L}$',color=colors[iL], linewidth=1.5, alpha = 0.7)
        if ifcsv:
            data_sets.append(tmp_list)
    
    if ifcsv:
        generate_csv('3', data_sets)
    ax = plt.gca()
    ticks = np.arange(0, 2.1, 0.1)
    ax.set_xticks(ticks)
    ax.tick_params(direction='in', which='both')
    
    plt.legend(loc='lower right',frameon=False,labelspacing=0.5)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.xlabel(r"$h$")
    plt.ylabel(r"$M$")
    ax.text(0.02, 0.98, region_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    
    ax_inset = inset_axes(ax, width="40%", height='40%', loc="upper right")
    hc(ax_inset)
    ax_inset.tick_params(axis='both', labelsize=8)
    
    ax_inset = inset_axes(ax, width="40%", height='40%', loc="lower left",bbox_to_anchor=(0.05, 0.04, 1, 1),bbox_transform=ax.transAxes,)
    zoomin(ax_inset)
    ax_inset.tick_params(axis='both', labelsize=8)
    
    #plt.show()
    plt.savefig(f"/Users/ruiqi/Documents/tmp/currents/fig3.png",dpi=300, bbox_inches='tight')

#store_hc()
marker_diffsize(ifcsv=True)
    