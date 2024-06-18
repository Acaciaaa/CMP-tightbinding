from __future__ import division
import kwant
import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import kron
import pylab as py
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import cmath
from math import sqrt, pi
import warnings

'''Defining Pauli matrices'''
s0 = np.array([[1,0],[0,1]], complex)
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]], complex)

class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

CONTINUEPOINT = 0.1

HALDANETRI, DEFECT = 'haldane and triangular', 'defect graphene'
NONTRIVIAL, TRIVIAL, NONE = 'nontrivial', 'trivial', 'none'
model = dict(name = NONE,  category = NONE,
         cc = 1.0, # cell-cell distance not site-site distance
         m = 1.0, t1 = 1/2, tc = 1/2, # on-site difference -> 2m, 1st neighbor -> t1, 2nd neighbor -> t2
         t2 = 1/2, m2 = 1/2, # nearest neighbor -> t2, on-site energy -> m2
         t3 = 1.0, # coupling between two lattices
         kappa = 1.0, deltaE = 1.0,
         L=4, W=4)

GAP, CHANGE = 'localizer gap', 'minimum eigenvalues change'
para = dict(func = NONE,
            y_min = 0, y_max = model['W'],
            x_min = 0, x_max = model['L'],
            num_cc = 50, num_eigvals = 10,
            y_fix = model['W']/2)

def model_builder():
    if model['name'] == DEFECT:
        return defect_graphene()
    return haldane_triangular()

def defect_graphene():
    cc = model['cc']
    lat = kwant.lattice.general([(cc, 0), (cc / 2, cc * sqrt(3) / 2)],
                                 [(0, 0), (0, cc / sqrt(3))],
                                 ['a', 'b'])
    a, b = lat.sublattices

    def onsite(site):
        if site.family.name == 'a':
            return m
        if site.family.name == 'b':
            return -m
        return 0

    L, W = model['L'], model['W']
    m = model['m']
    sys = kwant.Builder()
    def shape(pos):
        x, y = pos
        return (-(L-cc)/2 <= x <= (L+cc)/2) and ((-W/2+cc/(2*sqrt(3))) <= y <= (W/2-cc/sqrt(3)+cc*sqrt(3)/2))
    sys[lat.shape(shape, (0, 0))] = onsite

    t1, tc, = model['t1'], model['tc']

    neighbors_graphene = [(0, 0), (0, 1), (-1, 1)]
    for neighbor in neighbors_graphene:
        sys[kwant.builder.HoppingKind(neighbor, a, b)] = -t1

    temp = tc*cmath.exp((1.j)*pi/2.)

    neighbors_a = [(0, 0), (1, 0), (0, 1)]
    for sour, tar in zip(neighbors_a, neighbors_a[1:]+neighbors_a[:1]):
        sys[a(tar[0], tar[1]), a(sour[0], sour[1])] = temp
    neighbors_b = [(0, 0), (1, -1), (1, 0)]
    for sour, tar in zip(neighbors_b, neighbors_b[1:]+neighbors_b[:1]):
        sys[b(tar[0], tar[1]), b(sour[0], sour[1])] = temp

    kwant.plot(sys)
    return sys.finalized()

def haldane_triangular():
    cc = model['cc']
    lat = kwant.lattice.general([(cc, 0), (cc / 2, cc * sqrt(3) / 2)],
                                 [(0, 0), (0, cc / sqrt(3)), (cc / 2, cc / (2 * sqrt(3)))],
                                 ['a', 'b', 'c'])
    a, b, c = lat.sublattices

    m, m2 = model['m'], model['m2']
    def onsite(site):
        if site.family.name == 'c':
            return m2
        if site.family.name == 'a':
            return m
        return -m

    L, W = model['L'], model['W']
    if L == 0 and W == 0:
        sys = kwant.Builder()
        sys[lat.shape((lambda pos: True), (0, 0))] = onsite
    elif L == 0:
        sys = kwant.Builder(kwant.TranslationalSymmetry([cc, 0]))
        # y lower boundary attention!
        sys[lat.shape((lambda pos: -cc/(2*sqrt(3))<= pos[1] <=W), (0, 0))] = onsite
    else:
        sys = kwant.Builder()
        sys[lat.shape((lambda pos: (0<=pos[0]<=L) and (0<=pos[1]<= W)), (0, 0))] = onsite

    t1, tc, t2, t3 = model['t1'], model['tc'], model['t2'], model['t3']
    hoppings = [((0, 0), a, c), ((0, 0), b, c), ((1, 0), a, c), ((1, 0), b, c), ((0, 1), a, c), ((1, -1), b, c)]
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t3
    neighbors_triangular = [(0, 1), (-1, 0), (-1, 1)]
    for neighbor in neighbors_triangular:
        sys[kwant.builder.HoppingKind(neighbor, c, c)] = -t2
    neighbors_graphene = [(0, 0), (0, 1), (-1, 1)]
    for neighbor in neighbors_graphene:
        sys[kwant.builder.HoppingKind(neighbor, a, b)] = -t1

    temp = tc*cmath.exp((1.j)*pi/2.)
    neighbors_imag = [(0, 1), (-1, 0), (1, -1)]
    for neighbor in neighbors_imag:
        sys[kwant.builder.HoppingKind(neighbor, b, b)] = temp
        sys[kwant.builder.HoppingKind(neighbor, a, a)] = temp.conjugate()

    # kwant.plot(sys)
    # 2D PEC exception:
    if L == 0 and W == 0:
        return kwant.wraparound.wraparound(sys).finalized()
    return sys.finalized()

def spectral_localizer(sys, x, y, E):
    kappa = model['kappa']
    H = sys.hamiltonian_submatrix()
    dim = np.shape(H)[0]
    X, Y = np.zeros(H.shape), np.zeros(H.shape)
    for i, site in enumerate(sys.sites):
        X[i, i] = site.pos[0]
        Y[i, i] = site.pos[1]
    L = kron(sz, H) + kappa * (kron(sx, X-x*np.identity(dim)) + kron(sy, Y-y*np.identity(dim)))
    return L, dim

def localizer_gap(sys):
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
            L, dim = spectral_localizer(sys, x, y, 0)
            eigvals = eigsh(L, k=num_eigvals, which='SM', return_eigenvectors=False)
            min_eigvals[i][j] = np.min(np.abs(eigvals))

    plt.figure()
    plt.imshow(min_eigvals, cmap='viridis', interpolation='bicubic',
               extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]])
    plt.colorbar()
    plt.title('Spectral Localizer Minimum Eigenvalues')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    p_text = ', '.join([f'{key}: {value:.3f}' if isinstance(value, float) else f'{key}: {value}' for key, value in model.items()])
    plt.figtext(0.5, 0.01, p_text, ha="center", va="bottom", fontsize=10, color="blue", bbox=dict(facecolor='lightblue', edgecolor='blue'))
    filename = model['name'] + model['category']+f"t3={model['t3']:.2f}_tc={model['tc']:.2f}_t2={model['t2']:.2f}_m2={model['m2']:.2f}"
    plt.savefig(f'/content/localizer_gap_{filename}.png')
    plt.show()
    plt.close()

def eigenvalues_change(sys):
    cc=model['cc']
    num_cc, num_eigvals = para['num_cc'], para['num_eigvals']
    y_fix = para['y_fix']
    x_min, x_max = para['x_min'], para['x_max']
    num_x = int((x_max-x_min) *num_cc / cc)
    x_coords = np.linspace(x_min, x_max, num=num_x)
    tracked_eigvals = np.zeros((num_x, num_eigvals))

    for i, x in enumerate(x_coords):

        L, dim = spectral_localizer(sys, x, y_fix, 0)

        current_eigvals = eigsh(L, k=num_eigvals, which='SM', return_eigenvectors=False, tol=1e-5)

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
    for i in range(num_eigvals):
        line = tracked_eigvals[:, i]
        if np.max(np.abs(np.diff(line))) > CONTINUEPOINT:
            continue

        signs = np.sign(line)
        changes = np.diff(signs)
        crossing_indices = np.where(changes != 0)[0]

        for index in crossing_indices:
            #print(crossing_indices, x_coords[index], line[index], x_coords[index+1], line[index+1])
            zero_point.append((x_coords[index]+x_coords[index+1])/2)

        plt.plot(x_coords, line, label=f'Eig {i+1}')
    plt.xlabel('x')
    plt.ylabel('localizer eigenvalues')
    p_text = ', '.join([f'{key}: {value:.3f}' if isinstance(value, float) else f'{key}: {value}' for key, value in model.items()])
    plt.figtext(0.5, 0.01, p_text, ha="center", va="bottom", fontsize=10, color="blue")

    str_zero_point = 'zero point: ' + ', '.join([f'{item:.4f}' for item in zero_point])
    plt.figtext(0.5, 1, str_zero_point, ha="center", va="top", fontsize=10, color="blue")
    filename = model['name']+model['category']+f"t3={model['t3']:.2f}_tc={model['tc']:.2f}_t2={model['t2']:.2f}_m2={model['m2']:.2f}"
    plt.savefig(f'/content/eigenvalues_change_{filename}.png')
    plt.show()
    plt.close()

def change_model(name, category='none'):
    if name==HALDANETRI:
        if category==NONTRIVIAL:
            model.update(dict(name = name, category = category,
                        cc = 1.0,
                        m = 0.0, t1 = 1.0, tc = 0.5,
                        t2 = 0.2, m2 = -0.35,
                        t3 = 0.3,
                        kappa = 1.0,
                        L=10, W=10))
        elif category==TRIVIAL:
            model.update(dict(name = name, category = category,
                        cc = 1.0,
                        m = 2*sqrt(3), t1 = 1.0, tc = 0.5,
                        t2 = 0.2, m2 = -0.35,
                        t3 = 0.3,
                        kappa = 1.0,
                        L=10, W=10))
    elif name==DEFECT:
        model.update(dict(name=name, category = category,
                      cc = 1.0,
                      m = 0.0, t1 = 1.0, tc = 1.0,
                      kappa = 1.0,
                      L=5, W=6))
    else:
        print('change_model error')
        sys.exit()

def change_para(func):
    L, W, cc = model['L'], model['W'], model['cc']
    if func==GAP:
        if model['name'] == HALDANETRI: # 1/4 pic
            para.update(dict(func = func,
                             y_min = 0, y_max = W/2,
                             x_min = 0, x_max = L/2,
                             num_cc = 5, num_eigvals = 10))
        elif model['name'] == DEFECT: # whole pic
            para.update(dict(func = func,
                             y_min = -W/2+cc/(2*sqrt(3)), y_max = W/2-cc/sqrt(3)+cc*sqrt(3)/2,
                             x_min = -(L-cc)/2, x_max = (L+cc)/2,
                             num_cc = 5, num_eigvals = 10))
    elif func==CHANGE:
        if model['name'] == HALDANETRI: # middle y or real site
            para.update(dict(func = func,
                             x_min = 0, x_max = L/4, num_cc=200,
                             num_eigvals = 20,
                             y_fix = 3.5*cc/sqrt(3)))
        elif model['name'] == DEFECT: # exactly middle y
            para.update(dict(func = func,
                             x_min = 0, x_max = cc, num_cc=200,
                             num_eigvals = 20,
                             y_fix = cc*sqrt(3)/4 - cc/(4*sqrt(3))))
    else:
        print('change_para error')
        sys.exit()

def main_func(name, category):
    change_model(name, category)
    sys = model_builder()

    change_para(GAP)
    localizer_gap(sys)

    change_para(CHANGE)
    eigenvalues_change(sys)

def band_structure():
    change_model(HALDANETRI, NONTRIVIAL)
    
    for (t3, tc, t2, m2) in [(0.1, 0.1, 0.8, -0.1),
                         (0.1, 0.5, 1, -0.5),
                         (0.5, 0.1, 0.5, -0.1),
                         (1, 0.1, 0.1, -0.1),
                         (1, 0.1, 1, -0.5),
                         (1, 1, 1, -0.8)]:
        model['t3'], model['tc'], model['t2'], model['m2'] = t3, tc, t2, m2
        model['L'], model['W'] = 0, 10
        sys = model_builder()
        plt.figure()
        fig, ax = plt.subplots()
        kwant.plotter.bands(sys, momenta = np.linspace(0, 2*pi, 200), ax=ax)
        p_text = ', '.join([f'{key}: {value:.3f}' if isinstance(value, float) else f'{key}: {value}' for key, value in model.items()])
        plt.figtext(0.5, 0.01, p_text, ha="center", va="bottom", fontsize=10, color="blue", bbox=dict(facecolor='lightblue', edgecolor='blue'))
        filename = f"t3={model['t3']:.2f}_tc={model['tc']:.2f}_t2={model['t2']:.2f}_m2={model['m2']:.2f}"
        plt.savefig(f'/content/ribbon_{filename}.png')
        
def different_haldane():
    change_model(HALDANETRI, NONTRIVIAL)
    
    for (t3, tc, t2, m2) in [(0.1, 0.1, 0.8, -0.1),
                         (0.1, 0.5, 1, -0.5),
                         (0.5, 0.1, 0.5, -0.1),
                         (1, 0.1, 0.1, -0.1),
                         (1, 0.1, 1, -0.5),
                         (1, 1, 1, -0.8)]:
        model['t3'], model['tc'], model['t2'], model['m2'] = t3, tc, t2, m2
        sys = model_builder()
        
        # change_para(GAP)
        # localizer_gap(sys)

        change_para(CHANGE)
        eigenvalues_change(sys)

different_haldane()