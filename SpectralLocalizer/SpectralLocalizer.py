from __future__ import division
import kwant
from scipy.sparse.linalg import eigsh, eigs
from numpy.linalg import eigh
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
def custom_sort(evals, evecs):
    indices = np.argsort(np.abs(evals))
    sorted_evals = evals[indices]
    sorted_eigens = evecs[:, indices]
    return sorted_evals, sorted_eigens

class Localizer:
    def __init__(self, model):
        self.H, self.X, self.Y, self.dim = model.H, model.X, model.Y, model.dim
        self.H_part = np.kron(km.sz, self.H)
        self.X_part = np.kron(km.sx, self.X)
        self.Y_part = np.kron(km.sy, self.Y)
        
    def get_localizer(self, x=0, y=0, kappa=1):
        return self.H_part + kappa * (
            self.X_part + self.Y_part - np.kron(km.sx,x*np.identity(self.dim)) - np.kron(km.sy,y*np.identity(self.dim))
            )

def get_eigenvalues(name, *args):
    x_list, y_list, kappa_list, num_eigvals = args[0],args[1],args[2], args[3]
    sys = km.model_builder()
    if name == km.SSH:
        localizer = Localizer(SSH(sys, calculate_expectation=False))
    elif name == km.HALDANE:
        localizer = Localizer(HALDANE(sys, calculate_expectation=False))
        
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
        km.model['L']=km.model['W']=25
        h_list = np.array([0.2])
        # x y
        x_edge, y_edge = km.rectangle_vertex(km.model['L'], km.model['W'])
        num_cc = 50
        #num_x = int((x_edge-x_edge+2)*num_cc)
        num_x = int((x_edge-0)*num_cc)
        y_list=np.array([0])
        #x_list=np.linspace(x_edge-2,x_edge,num=num_x)
        x_list=np.linspace(0,x_edge,num=num_x)
        # kappa
        kappa_list = np.array([2.0])#0.01,0.1,0.5,1.0,2.0,3.0])
        # num_eigvals
        num_eigvals = 10
    elif name == km.SSH:
        # h (t1)
        km.change_model(km.SSH, km.NONE)
        km.model['L']=10
        h_list = np.array([0.5])
        # x
        num_cc = 50
        num_x = int(3*num_cc)
        x_list=np.linspace(0,3,num=num_x)
        y_list=np.array([0])
        # kappa
        kappa_list = np.array([0.01, 0.1, 0.5,1,3,5])
        # num_eigvals
        num_eigvals = 10
    
    for ih, h in enumerate(h_list):
        km.model['h'] = h
        results = get_eigenvalues(name,x_list,y_list,kappa_list,num_eigvals)
        for ikappa, kappa in enumerate(kappa_list):
            plt.figure()
            plt.axhline(0, color='grey', linewidth=1, alpha=0.4)
            for line in range(num_eigvals):
                plt.scatter(x_list, results[:, 0, ikappa, line],s=1)
            plt.title(rf"$\kappa={kappa}$", loc='left')
            plt.xlabel('x')
            plt.ylabel('localizer eigenvalues')
            plt.savefig(f"/Users/ruiqi/Documents/tmp/localizer/haldane/localizer_visualization/{kappa}.png", dpi=300, bbox_inches='tight')
            #plt.savefig(f"/Users/ruiqi/Documents/tmp/localizer/ssh/localizer_visualization/{kappa}.png", dpi=300, bbox_inches='tight')
            #plt.show()
            #plt.close()

class SSH:
    def __init__(self, sys, calculate_expectation=True):
        H = sys.hamiltonian_submatrix(sparse=False)
        self.H = H
        self.dim = np.shape(H)[0]
        self.l=self.dim-1
        self.X = np.zeros(H.shape)
        get_position_operator(sys, self.X, 0)
        self.Y = np.zeros(H.shape)
        evals, evecs = eigh(self.H)
        self.sorted_evals, self.sorted_evecs = custom_sort(evals, evecs)
        self.delta = abs(self.sorted_evals[0])
        
        if calculate_expectation:
            psiA = (self.sorted_evecs[:,0]-self.sorted_evecs[:,1])/sqrt(2)
            psiB = (self.sorted_evecs[:,0]+self.sorted_evecs[:,1])/sqrt(2)
            val1 = psiA.conj().T @ self.X @ psiA
            val2 = psiB.conj().T @ self.X @ psiB
            values = np.array([val1, val2])
            index = np.argmin(values)
            if index == 0:
                w, l_w = val1, val2
            else:
                w, l_w = val2, val1
                psiA, psiB = psiB, psiA
            assert np.allclose(np.imag(w), 0, atol=1e-10), "Warning: Data has non-negligible imaginary part!"
            self.psiA, self.psiB, self.w, self.l_w = psiA, psiB, np.real(w), np.real(l_w)
        
    def theoretical_w(self):
        ratio = km.model['h']/km.model['t2']
        Nc = km.model['L']
        denominator = (1-ratio**2)*(1-ratio**(2*Nc))
        numerator = ratio**2 + ratio**(2*Nc)*(-Nc+Nc*ratio**2-ratio**2)
        print(2*numerator/denominator, self.w)
    
    def first_order(self, kappa=1):
        numerator = self.delta
        x0_precision = self.l/2 - sqrt(-numerator**2/kappa**2 + (self.l/2-self.w)**2)
        x0_approximation = self.w + numerator**2/(self.l*kappa**2)
        print("theoretical first-order: ", x0_precision, x0_approximation)
        
    def calculate_u(self):
        even_indices = np.arange(0, len(self.sorted_evals), 2)
        eigenvalues = self.sorted_evals[even_indices]
        eigenstates = self.sorted_evecs[:, even_indices]
        u = 0.0
        u_n = np.zeros(len(eigenvalues)-1)
        for n in range(1, len(eigenvalues)):
            En = eigenvalues[n]
            psi_n = eigenstates[:, n]
            tmp =  (2/En) * (self.psiA.conj().T@self.X@psi_n) * (psi_n.conj().T@self.X@self.psiB)
            if np.isclose(tmp.imag, 0):
                term = tmp.real
            else:
                raise ValueError("tmp is complex")
            u_n[n-1]=abs(term)
            u+=term
        
        def draw_u():
            values = []
            values.append(np.max(u_n))
            for n, pct in enumerate([70]):
                threshold = np.percentile(u_n, pct)
                values.append(np.mean(u_n[u_n >= threshold]))
            correlation_coefficient, p_value = pearsonr(np.abs(eigenvalues[1:]), u_n)
        
        return u
    
    def second_order(self, kappa=1):
        u = self.calculate_u()
        numerator = self.delta + kappa**2*u
        x0_precision = self.l/2 - sqrt(-numerator**2/kappa**2 + (self.l/2-self.w)**2)
        x0_approximation = self.w + numerator**2/(self.l*kappa**2)
        print("theoretical second-order: ", x0_precision, x0_approximation)
        
class HALDANE:
    def __init__(self, sys, calculate_expectation=True):
        H = sys.hamiltonian_submatrix(sparse=False)
        self.H = H
        self.dim = np.shape(H)[0]
        self.L = km.model['L']
        self.X = np.zeros(H.shape)
        get_position_operator(sys, self.X, 0)
        self.Y = np.zeros(H.shape)
        get_position_operator(sys, self.Y, 1)
        
        a_list = [0.1, 0.5, 2, 5, 10]
        self.expectation = np.zeros((5, 6))
        self.expectation = np.array([
            [12.49089714, 11.97480348, 11.97480022, 10.94310137, 10.65326011, 10.94310144,],
            [12.49089722, 11.97480328, 11.97480002, 10.94310146, 10.65325867, 10.94310154,],
            [12.49126787, 11.97378827, 11.97378632, 10.94355909, 10.6465343,  10.94355913,],
            [12.49244111, 11.96498075, 11.96497814, 10.94571343, 10.61084394, 10.94571152,],
            [10.18463698,  9.77692345,  9.77202316,  8.93519283,  8.68369129,  8.9285602, ]])
        if calculate_expectation:
            x_positions, y_positions = np.diag(self.X), np.diag(self.Y)
            evals, evecs = eigh(self.H)
            energies, states = custom_sort(evals, evecs)
            for ia, a in enumerate(a_list):
                sigma = a/self.L
                gaussian_values = np.exp(-(energies-0)**2 / (2 * sigma**2))
                gaussian_values[energies>0] = 0
                gaussian_values /= np.sum(gaussian_values)
                for iarea, (axis, fixed_point) in enumerate([('x', 0.5/sqrt(3)), ('x', 1/sqrt(3)), ('x', 2/sqrt(3)), 
                                          ('y', 0), ('y', 0.5), ('y', 1)]):
                    if axis == 'x':
                        mask = (np.abs(y_positions - fixed_point) < 1e-3) & (x_positions > 0)
                    else:
                        mask = (np.abs(x_positions - fixed_point) < 1e-3) & (y_positions > 0)
                    P = np.diag(mask)

                    each_expectations = np.zeros(len(energies))
                    for i in range(len(energies)):
                        psi = states[:, i]
                        if axis == 'x':
                            numerator = np.vdot(psi, P @ self.X @ P @ psi)
                        else:
                            numerator = np.vdot(psi, P @ self.Y @ P @ psi)
                        denominator = np.vdot(psi, P @ psi)
                        each_expectations[i]=(np.real(numerator / denominator))
                    self.expectation[ia, iarea] = np.sum(gaussian_values * each_expectations)
        

def edgestate_location_1d():
    # h
    km.change_model(km.SSH, km.NONE)
    km.model['L']=10
    h_list = np.array([0.5])
    # kappa
    num_kappa = 200
    kappa_list = np.linspace(0.01, 2, num_kappa)
    
    for ih, h in enumerate(h_list):
        plt.figure()
        plt.grid(True, linestyle='--', alpha=0.4)
        km.model['h'] = h
        sys = km.model_builder()
        ssh = SSH(sys, calculate_expectation=True)
        localizer = Localizer(ssh)
        
        # TODO: 还有显著性的问题
        plt.axhline(ssh.w, color='red', linewidth=1,alpha=0.8)
        location_change = np.zeros(num_kappa)
        
        def local_chern_number(x):
            L = localizer.get_localizer(x=x, y=0, kappa=kappa)
            eigvals = eigsh(L, k=1, sigma=0, return_eigenvectors=False, tol=1e-5)
            return eigvals[0]
        
        for ikappa, kappa in enumerate(kappa_list):
            result = minimize_scalar(lambda x: abs(local_chern_number(x)), bounds=(0, 1), method='bounded', options={'xatol': 1e-10})
            location_change[ikappa] = result.x
        plt.scatter(kappa_list, location_change, s=2, alpha=0.6)
        plt.ylabel('location(x)')
        #plt.ylim(0, 0.5)
        plt.xlabel(r'$\kappa$')
        plt.show()
        #plt.savefig(f"/Users/ruiqixu/Desktop/kappa/localizer numerical/ssh/localizer location.png", dpi=300, bbox_inches='tight')
        plt.close()
        
def edgestate_location_2d():
    # h
    km.change_model(km.HALDANE, km.NOMASS)
    km.model['L']=km.model['W']=25
    h_list = np.array([0.2])
    # x y -> 2 areas
    x_edge, y_edge = km.rectangle_vertex(km.model['L'], km.model['W'])
    x_list=np.array([0, 0.5, 1])
    y_list=np.array([0.5/sqrt(3), 1/sqrt(3), 2/sqrt(3)])
    a_list = [0.1, 0.5, 2, 5, 10]
    # kappa
    num_kappa = 20
    kappa_list = np.linspace(0.01, 2, num_kappa)
    iarea = -1
    
    for axis, fixed_axis, fixed_list, edge_limit, edge_name in [('x', 'y', y_list, x_edge, 'right'),('y', 'x', x_list, y_edge, 'upper')]:
        for fixed_value in fixed_list:
            iarea += 1
            for ih, h in enumerate(h_list):
                plt.figure()
                plt.grid(True, linestyle='--', alpha=0.4)
                km.model['h'] = h
                sys = km.model_builder()
                haldane = HALDANE(sys, calculate_expectation=False)
                localizer = Localizer(haldane)
                for ia, a in enumerate(a_list):
                    tmp = haldane.expectation[ia, iarea]
                    plt.plot([0, 2], [tmp, tmp], linewidth=1, alpha=0.8, label=f'a={a}')
                plt.legend()
                plt.plot([0, 2], [edge_limit, edge_limit], linewidth=1, alpha=0.4, linestyle=':', color='black')

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
                    evals = np.linalg.eigvalsh(L)
                    return (np.count_nonzero(evals > 0) - np.count_nonzero(evals < 0)) / 2
            
                location_change = np.zeros(num_kappa)
                if iarea == 3:
                    location_change = np.load("/Users/ruiqi/Desktop/location_change_0.npy")
                #for ikappa, kappa in enumerate(kappa_list):
                 #   if local_chern_number(0,axis,kappa) == local_chern_number(edge_limit,axis,kappa):
                  #      location_change[ikappa] = 0
                   # location_change[ikappa] = binary_search(lambda v: local_chern_number(v, axis, kappa), 0, edge_limit, xtol=1e-3)
                #np.save(f"/Users/ruiqi/Documents/tmp/localizer/haldane/zero_kappa/location_change_{iarea}", location_change)
                #np.save(f"/storage/home/hcoda1/4/rxu366/p-ikimchi3-0/tmp/location_change_{iarea}", location_change)
                plt.scatter(kappa_list, location_change, s=2, alpha=0.6)
                plt.title(f"{fixed_axis}={fixed_value:.2f}", loc='left')
                plt.ylabel(axis)
                plt.xlabel(r'$\kappa$')
                plt.title(f"{edge_name} edge")
                plt.savefig(f"/Users/ruiqi/Documents/tmp/localizer/haldane/zero_kappa/{fixed_axis}_{fixed_value:.2f}_{km.model['L']}.png", dpi=300, bbox_inches='tight')
                #plt.savefig(f"/storage/home/hcoda1/4/rxu366/p-ikimchi3-0/tmp/{fixed_axis}_{fixed_value:.2f}_{km.model['L']}.png", dpi=300, bbox_inches='tight')
                #plt.show()
                #plt.close()
                if iarea == 3:
                    return
                
    
np.set_printoptions(suppress=True)
#eigenvalues_change(km.HALDANE)
edgestate_location_2d()
#edgestate_location_1d()

#km.change_model(km.SSH, km.NONE)
# km.change_model(km.HALDANE, km.NOMASS)
# km.model['L']=10
# km.model['W']=25
# km.model['h']=0.5
# sys=km.model_builder()
# ssh = SSH(sys, calculate_expectation=True)
# print(ssh.delta/(ssh.l/2-ssh.w))
















import time
def localizer_gap(sys, e=0):
    cc=model['cc']
    num_cc, num_eigvals = para['num_cc'], para['num_eigvals']

    x_min, x_max = para['x_min'], para['x_max']
    num_x = int((x_max-x_min) *num_cc / cc)
    y_min, y_max = para['y_min'], para['y_max']
    num_y = int((y_max-y_min) *num_cc / cc)

    x_coords = np.linspace(x_min, x_max, num=num_x)
    y_coords = (np.linspace(y_min, y_max, num=num_y))[::-1]
    min_eigvals = np.empty((num_y, num_x))
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            L, dim = spectral_localizer(sys, x=x, y=y, e=e)
            eigvals = eigsh(L, k=num_eigvals, sigma=0, return_eigenvectors=False)
            min_eigvals[i][j] = np.min(np.abs(eigvals))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.imshow(min_eigvals, cmap='viridis', interpolation='bicubic',
               extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]],
               vmin=0, vmax=1)
    plt.colorbar()
    kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    label = pick_label(model['name'])
    plt.title(label)
    plt.savefig(f'/content/localizer_gap_{label}.png')
    plt.show()
    plt.close()

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

def sigma_change():# precondition: eigenvalues_change already cross 0
    sigma_value = []
    h_value = np.linspace(0.8, 1.3, 100)
    for h in h_value:
        model['h'] = h
        sys = model_builder()
        L, dim = spectral_localizer(sys, 0, 0, 0)
        # only apply to haldane_defect because the smallest one may not be on the band crossing 0
        eigenvalue = eigsh(L, k=1, sigma=0, return_eigenvectors=False, tol=1e-5)
        sigma_value.append(eigenvalue[0])
        
    plt.plot(h_value, sigma_value)
    plt.axhline(0, color='red', linewidth=1.5, linestyle='--')
    plt.ylabel('sigma(origin) change')
    label = pick_label(model['name'], ifh=False)
    plt.figtext(0.5, 0.01, label, ha="center", va="bottom", fontsize=10, color="blue")
    
    f = interp1d(h_value, sigma_value)
    zero_1 = fsolve(f, x0=0.9)
    zero_2 = fsolve(f, x0=1.2)

    str_zero_point = f'zero crossing:{zero_1[0]:.3f}, {zero_2[0]:.3f}'
    plt.figtext(0.5, 0.8, str_zero_point, ha="center", va="top", fontsize=10, color="blue")
    plt.savefig(f'/content/sigma_change_{label}.png')
    plt.show()
    plt.close()
  
def eigenvalues_change(sys, e, *args):
    cc=model['cc']
    num_cc, num_eigvals = para['num_cc'], para['num_eigvals']
    if para['x_min'] == para['x_max']: # zigzag
        xlabel = f"x fixed at {para['x_min']:.2f}"
        coord_fix = para['x_min']
        v_min, v_max = para['y_min'], para['y_max']
    elif para['y_min'] == para['y_max']: # armchair
        xlabel = f"y fixed at {para['y_min']:.2f}"
        coord_fix = para['y_min']
        v_min, v_max = para['x_min'], para['x_max']
    num_coords = int((v_max-v_min) *num_cc / cc)
    v_coords = np.linspace(v_min, v_max, num=num_coords)
    tracked_eigvals = np.zeros((num_coords, num_eigvals))

    for i, y in enumerate(v_coords):
        if model['name'] == HALDANE:
            L, _ = spectral_localizer_1d(sys, y, e, *args)
        elif para['x_min'] == para['x_max']:
            L, _ = spectral_localizer_2d(sys, coord_fix, y, e, *args)
        elif para['y_min'] == para['y_max']:
            L, _ = spectral_localizer_2d(sys, y, coord_fix, e, *args)
        current_eigvals = eigsh(L, k=num_eigvals, sigma=0, return_eigenvectors=False, tol=1e-5)

        b = np.sort(current_eigvals)[::-1]
        
        if i == 0:
            tracked_eigvals[0, :] = b
            continue

        a = tracked_eigvals[i-1]
        if_newelement = np.max(np.abs(a-b))
        if if_newelement > CONTINUEPOINT:
            if a[0]-b[0] > 0:
                tracked_eigvals = np.roll(tracked_eigvals, -1, 1)
            else:
                tracked_eigvals = np.roll(tracked_eigvals, 1, 1)
        a = tracked_eigvals[i-1]
        for item in np.abs(a-b)[1:-1]:
            if item > CONTINUEPOINT:
                warnings.warn("Precision Issue!", UserWarning)
        tracked_eigvals[i, :] = b

    zero_point=[]
    plt.figure()
    plt.axhline(0, color='grey', linewidth=1)
    for i in range(num_eigvals):
        line = tracked_eigvals[:, i]
        # if np.max(np.abs(np.diff(line))) > CONTINUEPOINT:
        #     continue

        signs = np.sign(line)
        changes = np.diff(signs)
        crossing_indices = np.where(changes != 0)[0]

        for index in crossing_indices:
            #print(crossing_indices, x_coords[index], line[index], x_coords[index+1], line[index+1])
            zero_point.append((v_coords[index]+v_coords[index+1])/2)

        plt.scatter(v_coords, line, s=0.1, label=f'Eig {i+1}')
    plt.ylabel('localizer eigenvalues')
    label = pick_label(model['name'])
    plt.figtext(0.5, 0.01, label, ha="center", va="bottom", fontsize=10, color="blue")

    str_zero_point = 'zero point: ' + ', '.join([f'{item:.3f}' for item in zero_point])
    plt.figtext(0.5, 0.96, str_zero_point, ha="center", va="top", fontsize=10, color="blue")
    plt.figtext(0.5, 0.92, xlabel+f', e={e}', ha="center", va="top", fontsize=10, color="blue")
    plt.axis('equal')
    plt.savefig(f'/content/eigenvalues_change_{label}_{e}_{xlabel}.png')
    plt.show()
    plt.close()

def signature_change3(sys, e, bounds):
    import adaptive
    def find_signature(point):
        x, y = point
        L, dim = spectral_localizer(sys, x=x, y=y, e = e)
        evals, evecs = eigh(L)
        filtered_evals = evals[~np.isclose(evals, 0)]
        pos = np.sum(filtered_evals > 0)
        neg = np.sum(filtered_evals < 0)
        return (pos - neg)/2

    learner = adaptive.Learner2D(find_signature, bounds=bounds)
    initial_points = [(0, 0), (-2.4, -2.4), (-1.5, -0.5), (-1.5, 0.5), (1.5, -0.5), (1.5, 0.5), (0, 1.5), (0, -1.5)]
    for point in initial_points:
        learner.tell(point, find_signature(point))

    adaptive.runner.simple(learner, goal=lambda l: l.loss() < 0.05)

    data = learner.data
    X, Y, Z = zip(*[(x, y, z) for (x, y), z in data.items()])

    # 绘制结果
    # fig, ax = plt.subplots()
    plt.tricontourf(X, Y, Z, 100, cmap='RdYlGn', vmin=-1.01, vmax=1.01)  # 使用三角网格绘制等高线填充
    # kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
    plt.colorbar()
    # plt.xlim(-3, 3)
    # plt.ylim(-3, 3)
    
    # label = pick_label(model['name'])
    # plt.title(label+f'_e={e}')
    # plt.savefig(f'/content/signature_adaptive_{label}_{e}.png')
    # plt.show()
    # plt.close()

from matplotlib.colors import ListedColormap, Normalize
import time
def signature_change2(sys, e):
    def find_signature(x, y):
        L, dim = spectral_localizer(sys, x, y, e = e)
        evals, evecs = eigh(L)
        filtered_evals = evals[~np.isclose(evals, 0)]
        pos = np.sum(filtered_evals > 0)
        neg = np.sum(filtered_evals < 0)
        return (pos - neg)/2

    x = np.linspace(-3, 0, 48)
    y = np.linspace(-3, 0, 48)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(find_signature)(X, Y)

    cmap = ListedColormap(['lightgreen', 'white', 'pink', 'red'])
    norm = Normalize(vmin=-1.5, vmax=2.5)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap=cmap, norm=norm)
    plt.xlim(-4, 0)
    plt.ylim(-4, 0)
    kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
    plt.colorbar()
    
    label = pick_label(model['name'])
    plt.title(label+f'_e={e}')
    plt.savefig(f'/content/signature_grid_{label}_{e}.png')
    plt.show()

def signature_change(n = None, e = 0):
    signature_value = []
    h_value = np.linspace(0.3, 1.3, 1000)
    for h in h_value:
        model['h'] = h
        sys = model_builder()
        if n is not None:
            H = sys.hamiltonian_submatrix(sparse=False)
            evals, evecs = eigh(H)
            e = abs(sorted(evals, key=abs)[2*n])
        L, dim = spectral_localizer(sys, 0, 0, e)
        evals, evecs = eigh(L)
        filtered_evals = evals[~np.isclose(evals, 0)]
        pos = np.sum(filtered_evals > 0)
        neg = np.sum(filtered_evals < 0)
        signature_value.append((pos - neg)/2)
        
    plt.scatter(h_value, signature_value)
    plt.axhline(0, color='grey', linewidth=1)
    plt.ylabel('local chern number(origin) change')
    label = pick_label(model['name'], ifh=False)
    plt.figtext(0.5, 0.01, label+f'_e={e:.2f}', ha="center", va="bottom", fontsize=10, color="blue")
    
    changes = np.where(np.diff(signature_value) != 0)[0]
    changes_str = 'sign change: ' + ', '.join(f"{h_value[change]:.3f}" for change in changes)
    plt.figtext(0.5, 0.8, changes_str, ha="center", va="top", fontsize=10, color="blue")
    plt.savefig(f'/content/signature_change_origin_{label}_{e:.2f}.png')
    plt.show()
    plt.close()


def band_structure():
    change_model(HALDANETRI, NONTRIVIAL)

    # for t3 in [0.1, 3]:
    #     for t2 in [0.1, 3]:
    #         for tc in [0.1, 3]:
    #             model['t3'], model['tc'], model['t2'] = t3, tc, t2
    #             model['L'], model['W'] = 0, 11
    model['L'] = 0
    sys = model_builder()
    plt.figure()
    fig, ax = plt.subplots()
    kwant.plotter.bands(sys, momenta = np.linspace(0, 2*pi, 200), ax=ax)
    label = pick_label(model['name'])
    plt.figtext(0.5, 0.01, label, ha="center", va="bottom", fontsize=10, color="blue", bbox=dict(facecolor='lightblue', edgecolor='blue'))
    plt.show()
    plt.savefig(f'/content/ribbon_{label}.png')
    plt.close()

def different_haldane():
    change_model(HALDANETRI, NONTRIVIAL)
    for t3 in [0.1, 3]:
        for t2 in [0.1, 3]:
            for h in [0.1, 3]:
                  model['t3'], model['h'], model['t2'] = t3, h, t2
                  sys = model_builder()

                  change_para(GAP)
                  localizer_gap(sys)

                  # change_para(CHANGE)
                  # eigenvalues_change(sys)

# band_structure()
#different_haldane()
# sync_png_files()

#current_Jr(DEFECT, SINGLE, max_E=0.175)

def draw_H_psi(sys, state, x_range):
    J = kwant.operator.Current(sys)
    H, X, Y, dim = HXY(sys)
    evals, evecs = eigh(H)
    def custom_sort(evals, evecs):
        indices = np.argsort(np.abs(evals))
        sorted_evals = evals[indices]
        sorted_eigens = evecs[:, indices]
        return sorted_evals, sorted_eigens
    sorted_evals, sorted_evecs = custom_sort(evals, evecs)
    def draw_sth():
        evec = sorted_evecs[:, state*2]
        # current = J(evec)
        # fig, ax = plt.subplots()
        # kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
        # kwant.plotter.current(sys, current, ax=ax, colorbar=True)
        # plt.show()
        data = np.abs(evec)**2
        vmax = np.max(data)
        print(vmax)
        fig, ax = plt.subplots()
        kwant.plotter.map(sys, data, vmax=vmax, ax=ax)
        kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
        plt.show()
    def calculate_expectation():
        psi = sorted_evecs[:, state*2]
        P = np.zeros(H.shape)
        for i, site in enumerate(sys.sites):
            if site.pos[1] > 0:
                P[i, i] = 0
            else:
                P[i, i] = 1
        Px = np.zeros(H.shape)
        for i, site in enumerate(sys.sites):
            if site.pos[0]>=x_range[0] and site.pos[0]<=x_range[1]:
                Px[i, i] = 1
            else:
                Px[i, i] = 0
        psi_project = Px@P@psi
        psi_normalized = psi_project / np.sqrt(np.conj(psi_project).T @ psi_project)
        expectation = np.conj(psi_normalized).T @ Y @ psi_normalized
        return sorted_evals[state*2], expectation
    #draw_sth()
    return calculate_expectation()
            
def draw_L_psi(sys, x, y, e, *args):
    def draw_tool(v, title):
        data = np.abs(v)**2
        vmax = np.max(data)
        # print(vmax)
        fig, ax = plt.subplots()
        kwant.plotter.map(sys, data, vmax=vmax, ax=ax)
        ax.set_title(title)
        plt.show()
    
    def subvec(current_evec):
        psi11, psi12, psi21, psi22 = current_evec[:(dim//2), 0], current_evec[(dim//2):dim, 0], current_evec[dim:(3*dim//2), 0], current_evec[(3*dim//2):, 0]
        # print('\npsi11: ', psi11)
        # print('\npsi21: ', psi21)
        # print('\npsi12: ', psi12)
        # print('\npsi22: ', psi22)
        # print('\nmax abs:')
        # for psi in [psi11, psi12]: #, psi21, psi22, psi1, psi2]:
        #     print(np.max(np.abs(psi)))
        return psi11, psi12, psi21, psi22
    
    np.set_printoptions(precision=4, suppress=True)
    np.set_printoptions(linewidth=160)
    L, dim = spectral_localizer_edit(sys, x, y, e, *args)
    np.random.seed(0)
    eval, evec = eigsh(L, k=1, which='SM', return_eigenvectors=True)#, tol=1e-10)
    psi1, psi2 = evec[:dim, 0], evec[dim:, 0]
    #psi11, psi12, psi21, psi22 = subvec(evec)
    print('eval: ', eval)
    print('lattice number: ', dim)
    print('<1|2>: ', np.vdot(psi1, psi2))# +np.vdot(evec[n:, 0], evec[:n, 0]))
    print('<1|1>: ', np.vdot(psi1, psi1), '<2|2: >', np.vdot(psi2, psi2))

    def adjust_phase(vec):
        idx = np.argmax(np.abs(vec))
        print('max_idx: ', idx)
        phase = np.exp(-1j * np.angle(vec[idx]))
        return phase
    
    def phase_difference(angle_original, angle_adjust):
        phase_differences = angle_adjust-angle_original
        normalized_phase_differences = (phase_differences + np.pi) % (2 * np.pi) - np.pi
        print('phase differences: ', np.degrees(normalized_phase_differences)[0])

    angle_original = np.angle(evec[:, 0])
    def find_difference():
        psi21_predict = psi11.imag + psi11.real * 1j
        psi22_predict = -psi12.imag - psi12.real * 1j
        #print(psi21 - psi21_predict)
        #print(psi22 - psi22_predict)
        # draw_tool(np.concatenate((psi21-psi21_predict, psi22-psi22_predict)), 'predict')
        # print(np.abs(psi2)-np.abs(psi1))
        phase1 = np.angle(psi1)
        phase2 = np.angle(psi2)
        phase_difference = phase2 - phase1
        phase_difference = (phase_difference + np.pi) % (2 * np.pi) - np.pi
        print("相位差：", np.degrees(phase_difference))

    evec[:, 0] *= adjust_phase(psi1)
    psi1, psi2 = evec[:dim, 0], evec[dim:, 0]
    def verify_assume():
        def format_complex(c, decimals=3):
            return f"{c.real:.{decimals}f} + {c.imag:.{decimals}f}j"
        kappa= model['kappa']
        U = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            U[i, i] = psi2[i] / psi1[i]
        Udagger = np.conj(U)
        H, X, Y, _ = HXY(sys)
        P = X-x*np.identity(dim) - 1j * (Y-y*np.identity(dim))
        P_diag = np.diag(P)
        P_diag_edit = [(z / np.abs(z) * (np.abs(z) ** args[1])) if np.abs(z) != 0 else 0 for z in P_diag]
        P_edit = np.diag(P_diag_edit)
        P_rot = P_edit@U
        #print(P_rot)

        def plot_dynamics():
            PU = [P_rot[i, i] for i in range(dim)]
            pos = [sys.sites[i].pos for i in range(dim)]
            probabilities = np.abs(psi1)
            sorted_indices = np.argsort(probabilities)
            sorted_indices = sorted_indices[::-1]
            sorted_psi1 = psi1[sorted_indices]
            sorted_pos = np.array(pos)[sorted_indices]
            sorted_PU = (np.array(PU))[sorted_indices]
            for i in range(5):
                print(sorted_psi1[i], sorted_pos[i])
            # 计算辐角和模
            angles = np.angle(sorted_PU)
            magnitudes = np.abs(sorted_PU)  # 可用于调整箭头长度
            print(magnitudes)
            fig, ax = plt.subplots()
            for i in range(17):
                (pos_x, pos_y), angle, mag = sorted_pos[i], angles[i], magnitudes[i]
                ax.quiver(pos_x, pos_y, mag * np.cos(angle), mag * np.sin(angle), scale=10, color='blue')
            ax.axis('equal')  # 确保x和y轴的比例相同，避免箭头变形
            ax.set_xlim(-2, 2)
            ax.set_ylim(-4.2, -2)
            kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
            plt.show()
        plot_dynamics()
    verify_assume()

    # angle_max = np.angle(evec[:, 0])
    # phase_difference(angle_original, angle_max)
    # evec[:, 0] *= adjust_phase(psi11)
    # phase_difference(angle_max, np.angle(evec[:, 0]))
    psi11, psi12, psi21, psi22 = subvec(evec)

    def current_relation(psi, ifdirect):
        H, _, _, _ = HXY(sys)
        rho = np.outer(psi, np.conj(psi))
        if ifdirect:
            J = (H * rho - np.conj(H) * rho.T).imag
        else:
            J = (H @ rho - np.conj(H) @ rho.T).imag
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.xlim(-2, 2)
        plt.ylim(-4.1, -2.5)
        #plt.ylim(-2, 2)
        
        for i, j in sys.graph:
            weight = J[i][j]
            if weight < 1e-3:
                continue
            start_point, end_point = sys.sites[i].pos, sys.sites[j].pos
            normalized = (end_point-start_point)/np.linalg.norm(end_point - start_point)
            arrow_length = normalized * weight * 3
            arrow = patches.FancyArrowPatch(start_point, start_point+arrow_length,
                                            arrowstyle='-|>', connectionstyle='arc3,rad=0.0', mutation_scale=10, color='blue')
            ax.add_patch(arrow)
        kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
        plt.show()
        plt.close()
        return J
    J1 = current_relation(psi1, ifdirect=False)
    J2 = current_relation(psi2, ifdirect=False)
    

    def calculate_expectation():
        H, X, Y, _ = HXY(sys)
        psi1_H = np.conj(psi1).T @ (H-e*np.identity(dim)) @ psi1
        psi2_H = np.conj(psi2).T @ (H-e*np.identity(dim)) @ psi2
        psi1_X = np.conj(psi1).T @ (X-x*np.identity(dim)) @ psi1
        psi2_X = np.conj(psi2).T @ (X-x*np.identity(dim)) @ psi2
        psi1_Y = np.conj(psi1).T @ (Y-y*np.identity(dim)) @ psi1
        psi2_Y = np.conj(psi2).T @ (Y-y*np.identity(dim)) @ psi2
        print(psi1_H.real, psi2_H.real, psi1_X.real, psi2_X.real, psi1_Y.real, psi2_Y.real)
        
        #print((H-e*np.identity(dim)) @ psi1)
        #print((H-e*np.identity(dim)) @ psi2)
    calculate_expectation()

    def draw_psi():
        draw_tool(psi1, 'psi1')
        draw_tool(psi2, 'psi2')

        H, X, Y, _ = HXY(sys)
        draw_tool(H@psi1, 'H@psi1')
        print('H: ', H@psi1)
        
        draw_tool((X-x*np.identity(dim))@psi2, 'X-x@psi2')
        print('X-x: ', (X-x*np.identity(dim))@psi2)
        draw_tool((Y-y*np.identity(dim))@psi2, 'Y-y@psi2')
        print('Y: ', (Y-y*np.identity(dim))@psi2)

from scipy.optimize import brentq
def analyze_theory_ssh():
    change_model(SSH, NONE)
    model['L'] = 3
    sys = model_builder()
    def calculate_expectation():
        H = sys.hamiltonian_submatrix(sparse=False)
        print(H)
        dim = np.shape(H)[0]
        X = np.zeros(H.shape)
        for i, site in enumerate(sys.sites):
            X[i, i] = site.pos[0]
        evals, evecs = eigh(H)
        def custom_sort(evals, evecs):
            indices = np.argsort(np.abs(evals))
            sorted_evals = evals[indices]
            sorted_eigens = evecs[:, indices]
            return sorted_evals, sorted_eigens
        sorted_evals, sorted_evecs = custom_sort(evals, evecs)
        print(sorted_evals[0], sorted_evecs[0])
        print(sorted_evals[1], sorted_evecs[1])
        # print(np.conj(sorted_evecs[0]).T@X@sorted_evecs[0])
    calculate_expectation()

    def adjust_abs():
        def find_crossing(a, b, x_num):
            x_list = np.linspace(a, b, x_num)
            eigenvalue_list = np.zeros(x_num)
            for i, x in enumerate(x_list):
                L, _ = spectral_localizer_1d(sys, x, 0, 'abs', n)
                eigenvalue_list[i] = abs(eigsh(L, k=1, sigma=0, return_eigenvectors=False, tol=1e-5)[0])
            min_index = np.argmin(eigenvalue_list)
            min_value = eigenvalue_list[min_index]
            return x_list[min_index]
            
        plt.figure()
        x_values = []
        for n in [0.5, 1, 2, 2.5]:
            kappa_list = np.linspace(0.1, 2, 40)
            pos_list = np.zeros((len(kappa_list)))
            for i, kappa in enumerate(kappa_list):
                model['kappa'] = kappa
                a, b = 0, 1.2
                zero = find_crossing(a, b, 60)
                pos_list[i] = zero
            x_values.append(pos_list)
            plt.plot(kappa_list, pos_list, label=n)
        x_values = np.array(x_values)
        std_devs = np.std(x_values[1:], axis=0)
        min_std_index = np.argmin(std_devs)
        min_std_x = kappa_list[min_std_index]
        min_std_y = np.sum(x_values[1:, min_std_index])/4
        plt.legend()
        plt.xlabel('kappa')
        plt.ylabel('position')
        label = pick_label(model['name'], ifkappa=False)
        plt.figtext(0.5, 0.96, label, ha="center", va="top", fontsize=10, color="blue")
        #plt.figtext(0.5, 0.92, f"x: {x}, e: {e:.3f}, expectation: {expectation.real:.3f}, kappa: {min_std_x:.3f}, pos: {min_std_y:.3f}", ha="center", va="top", fontsize=10, color="blue")
        plt.show()
    #adjust_abs()

def analyze_theory_haldane(e_offer = False):
    change_model(HALDANE, NOMASS)
    model['L'], model['W'] = 9, 9
    model['h'] = 1
    sys = model_builder()
    #draw_L_psi(sys, 0, -3.5, 0, 'abs', 0)
    l, w = change_para(CHANGE)
    x = 0
    #for x_range in [[-0.2, 0.2], [-1.2, 1.2], [-2.2, 2.2], [-l-0.1, l+0.1]]:
    e, expectation = draw_H_psi(sys, state=0, x_range=[-3.2, 3.2])
    #print(x_range, f'{expectation.real:.3f}')
    if e_offer == False:
        e = 0
    def check_eigenvalues():
        change_para(CHANGE)
        para['x_min'] = para['x_max'] = x
        
        model['kappa'] = 1.5
        for n in [0, 0.5, 1, 2, 2.5]:
            eigenvalues_change(sys, e, 'abs', n)
    def adjust_abs():
        def brentq_search(y):
            L, dim = spectral_localizer_edit(sys, x, y, e, 'abs', n)
            return eigsh(L, k=1, sigma=0, return_eigenvectors=False, tol=1e-5)
        def f(y):
            L, dim = spectral_localizer_edit(sys, x, y, e, 'abs', n)
            evals, evecs = eigh(L)
            filtered_evals = evals[~np.isclose(evals, 0)]
            pos = np.sum(filtered_evals > 0)
            neg = np.sum(filtered_evals < 0)
            return ((pos - neg)/2)
        def binary_search(f, low, high, xtol):
            while (high - low) > xtol:
                mid = (low + high) / 2
                if f(mid) == 1 and f(low) == 0:
                    high = mid  # 如果mid为1，缩小区间到左半部分
                else:
                    low = mid  # 如果mid为0，缩小区间到右半部分
            return (low + high) / 2  # 返回区间中点作为跳变点的估计

        plt.figure()
        y_values = []
        for n in [0, 0.5, 1, 2, 2.5]:
            kappa_list = np.linspace(0.1, 2, 40)
            pos_list = np.zeros((len(kappa_list)))
            for i, kappa in enumerate(kappa_list):
                model['kappa'] = kappa
                a, b = -w, 0
                # if n != 0 or n!=2:
                #     zero = brentq(brentq_search, a, b, xtol=1e-5)
                # else:
                zero = binary_search(f, a, b, xtol=1e-5)
                pos_list[i] = zero
            y_values.append(pos_list)
            plt.plot(kappa_list, pos_list, label=n)
        y_values = np.array(y_values)
        std_devs = np.std(y_values[1:], axis=0)
        min_std_index = np.argmin(std_devs)
        min_std_x = kappa_list[min_std_index]
        min_std_y = np.sum(y_values[1:, min_std_index])/4
        plt.legend()
        plt.xlabel('kappa')
        plt.ylabel('position')
        label = pick_label(model['name'], ifkappa=False)
        plt.figtext(0.5, 0.96, label, ha="center", va="top", fontsize=10, color="blue")
        plt.figtext(0.5, 0.92, f"x: {x}, e: {e:.3f}, expectation: {expectation.real:.3f}, kappa: {min_std_x:.3f}, pos: {min_std_y:.3f}", ha="center", va="top", fontsize=10, color="blue")
        plt.show()
    #adjust_abs()
    check_eigenvalues()
    
#analyze_theory()

def perturbation_position(sys, x, y, epsilon_x, epsilon_y):
    L, dim = spectral_localizer(sys, x=x, y=y, e=0)
    evals, evecs= eigh(L)
    sorted_evals, sorted_evecs = custom_sort(evals, evecs)
    psi = sorted_evecs[:, 0]
    Lprime, _ = spectral_localizer(sys, x=(x+epsilon_x), y=(y+epsilon_y), e=0)
    V = Lprime - L

    first_order = np.conj(psi).T @ V @ psi

    second_order = 0
    for i, m in enumerate((sorted_evecs[:, 1:]).T):
        m_to_psi = np.conj(m) @ V @ psi.T
        correction_term = (m_to_psi.conjugate() * m_to_psi) / (0 - sorted_evals[i+1])
        second_order += correction_term

    third_order = 0
    # for i, m in enumerate((sorted_evecs[:, 1:]).T):
    #     for j, n in enumerate((sorted_evecs[:, 1:]).T):
    #         psi_to_m = np.conj(psi) @ V @ m.T
    #         m_to_n = np.conj(m) @ V @ n.T
    #         n_to_psi = np.conj(n) @ V @ psi.T
    #         correction_term = (psi_to_m*m_to_n*n_to_psi) / ((0 - sorted_evals[i+1])*(0 - sorted_evals[j+1]))
    #         third_order += correction_term
    
    print('perturbation: ', first_order, second_order)

from matplotlib.patches import Circle
from scipy.stats import pearsonr
def adjust_r(buckets):
    change_model(DEFECT, SINGLE)
    sys = model_builder()
    fig, ax = plt.subplots()
    kwant.plot(sys, ax=ax)
    for r in buckets:
        circle = Circle((0, 0), r, edgecolor='r', facecolor='none', linewidth=1)
        ax.add_patch(circle)
    ax.set_aspect('equal')
# adjust_r([0.6, 1.6, 3, 5])

#current_Jr(DEFECT, SINGLE)

def compute_u(H, X):
    eigenvalues, eigenstates = eigh(H)
    sorted_indices = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[sorted_indices]
    eigenstates = eigenstates[:, sorted_indices]
    psi1 = eigenstates[:, 0]
    psi2 = eigenstates[:, 1]
    psi1_prime = (psi1 + psi2) / np.sqrt(2)
    psi2_prime = (psi1 - psi2) / np.sqrt(2)
    even_indices = np.arange(0, len(eigenvalues), 2)
    eigenvalues = eigenvalues[even_indices]
    eigenstates = eigenstates[:, even_indices]
    
    total_sum = 0.0
    data = np.zeros(len(eigenvalues)-1)
    for n in range(1, len(eigenvalues)):
        En = eigenvalues[n]
        psi_n = eigenstates[:, n]
        tmp =  (2/En) * (np.vdot(psi1_prime, X @ psi_n)) * (np.vdot(psi_n, X @ psi2_prime))
        if np.isclose(tmp.imag, 0):
            term = tmp.real
        else:
            raise ValueError("tmp is complex")

        data[n-1]=abs(term)
        total_sum+=term
    
    values = []
    values.append(np.max(data))
    for n, pct in enumerate([70]):
        threshold = np.percentile(data, pct)
        values.append(np.mean(data[data >= threshold]))
        
    correlation_coefficient, p_value = pearsonr(np.abs(eigenvalues[1:]), data)
    
    return total_sum, values[0], values[1], correlation_coefficient, p_value

def draw_u():
    change_model(SSH, NONE)
    num = 67
    L_list = range(3, 3+num)
    ur_list, u_list = np.zeros((num, 2)),np.zeros(num)
    corr_list, p_list = np.zeros(num), np.zeros(num)
    for n, L in enumerate(L_list):
        model['L'] = L
        sys = model_builder()
        H = sys.hamiltonian_submatrix(sparse=False)
        X = np.zeros(H.shape)
        for i, site in enumerate(sys.sites):
            X[i, i] = site.pos[0]
        u_list[n], ur_list[n,0], ur_list[n,1], corr_list[n], p_list[n] = compute_u(H, X)
    
    plt.figure()
    plt.plot(L_list, corr_list, c='dimgray', label=f'correlation coefficient')
    plt.plot(L_list, p_list, c='#4B0082', label=f'p value')
    plt.axhline(0, color='gray', linestyle='-', linewidth=1)
    plt.axhline(-1, color='gray', linestyle='-', linewidth=1)
    plt.xlabel('Nc')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.scatter(L_list, ur_list[:,0], s=3, label='largest')
    plt.scatter(L_list, ur_list[:,1], s=3, label='70th pct.')
    plt.scatter(L_list, np.abs(u_list), s=3, label='|u|')
    plt.axhline(0, color='gray', linestyle='-', linewidth=1)
    plt.xlabel('Nc')
    
    plt.legend()
    plt.show()
#draw_u()