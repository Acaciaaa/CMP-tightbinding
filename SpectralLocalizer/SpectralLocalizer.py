from __future__ import division
import kwant
import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from numpy.linalg import eigh
from scipy.linalg import kron
import pylab as py
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys as system
import cmath
from math import sqrt, pi, sin, cos
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

HALDANETRI, DEFECT, PYBINDING = 'haldane and triangular', 'defect graphene', 'pybinding'
NONTRIVIAL, TRIVIAL, SINGLE, CLUSTER, NONE = 'nontrivial', 'trivial', 'single', '7 hexagons', 'none'
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
            num_cc = 50, num_eigvals = 10)

def model_builder():
    if model['name'] == DEFECT:
        return defect_graphene()
    if model['name'] == HALDANETRI:
        return haldane_triangular()
    # return haldane_triangular_pybinding()

def defect_graphene():
    cc = model['cc']
    lat = kwant.lattice.general([(cc, 0), (cc / 2, cc * sqrt(3) / 2)],
                                 [(cc/2, -cc / (2*sqrt(3))), (cc/2, cc / (2 * sqrt(3)))],
                                 ['a', 'b'], norbs=1)
    a, b = lat.sublattices

    m = model['m']
    def onsite(site):
        if site.family.name == 'a':
            return m
        if site.family.name == 'b':
            return -m
        return 0

    L, W = model['L'], model['W']
    (x, y) = rectangle_vertex(L, W)
    sys = kwant.Builder()
    sys[lat.shape((lambda pos: (-x<=pos[0]<=x) and (-y<=pos[1]<= y)), (0, 0))] = onsite

    t1, tc, = model['t1'], model['tc']

    neighbors_graphene = [(0, 0), (0, -1), (1, -1)]
    for neighbor in neighbors_graphene:
        sys[kwant.builder.HoppingKind(neighbor, b, a)] = -t1

    temp = tc*cmath.exp((1.j)*pi/2.)
    category = model['category']
    
    if category == SINGLE:
        neighbors_a = [(0, 0), (-1, 1), (-1, 0)]
        for sour, tar in zip(neighbors_a, neighbors_a[1:]+neighbors_a[:1]):
            sys[a(tar[0], tar[1]), a(sour[0], sour[1])] = temp
        neighbors_b = [(0, 0), (-1, 0), (0, -1)]
        for sour, tar in zip(neighbors_b, neighbors_b[1:]+neighbors_b[:1]):
            sys[b(tar[0], tar[1]), b(sour[0], sour[1])] = temp
        
    elif category == CLUSTER:
        mark = [0, -1, -2]
        for row_index, a2 in enumerate(mark):
            for column_index, a1 in enumerate(mark):
                if row_index == column_index == 0 or row_index == column_index == 2:
                    continue
                neighbors_b = [(a1+1, a2), (a1+1, a2+1), (a1, a2+1)]
                for sour, tar in zip(neighbors_b, neighbors_b[1:]+neighbors_b[:1]):
                    sys[b(tar[0], tar[1]), b(sour[0], sour[1])] = temp
                neighbors_a = [(a1, a2+2), (a1, a2+1), (a1+1, a2+1)]
                for sour, tar in zip(neighbors_a, neighbors_a[1:]+neighbors_a[:1]):
                    sys[a(tar[0], tar[1]), a(sour[0], sour[1])] = temp
    else:
        system.exit()
        
    # kwant.plot(sys)
    return sys.finalized()
    
def current_Jr(name, category, max_E = 0.5):
    max_E_label = 'max energy: '+ f'{max_E:.3f}'
    
    def edge_info(sys):
        distance = {}
        pos_info = [] # [[dot1, r1], [dot2, r2]]
        r_index = 0
        for head, tail in sys.graph:
            p1, p2 = sys.sites[head].pos, sys.sites[tail].pos
            unit_vector = (p2-p1)/np.linalg.norm(p2-p1)
            p3 = (p1+p2)/2
            theta = np.arctan2(p3[1], p3[0])
            thetahat = [-sin(theta), cos(theta)]
            
            r = round(np.linalg.norm(p3), 3)
            if r not in distance:
                distance[r] = r_index
                r_index += 1
            pos_info.append([np.dot(unit_vector, thetahat), r])
        return distance, pos_info
    
    def current_info(sys):
        H = sys.hamiltonian_submatrix(sparse=False)
        J = kwant.operator.Current(sys)
        evals, evecs = eigh(H)
        sum_current = None
        for i, e in enumerate(evals):
            if abs(e) < max_E:
                current = J(evecs[:, i])
                if sum_current is None:
                    sum_current = current
                else:
                    sum_current += current
        return sum_current
    
    def magnitude_info():
        temp = [None] * len(distance)
        magnitude = np.zeros(len(distance))
        for index, current in enumerate(sum_current):
            result = current * pos_info[index][0]
            r = pos_info[index][1]
            if temp[distance[r]] is None:
                temp[distance[r]] = [result]
            else:
                temp[distance[r]].append(result)
        for key, value in distance.items():
            magnitude[value] = np.sum(temp[value])
        return magnitude
    
    change_model(name, category)
    temp_sys = model_builder()
    distance, pos_info = edge_info(temp_sys)
    sort_r = sorted(distance.keys())
    num_tc = 500
    data = np.zeros((num_tc, len(distance)))
    tc_list = np.linspace(0.8, 1.3, num_tc)
    for index, tc in enumerate(tc_list):
        model['tc'] = tc
        sys = model_builder()
        sum_current = current_info(sys)
        magnitude = magnitude_info()
        data[index, :] = magnitude
    # r最小为例子
    label = pick_label(model['name'], iftc = False)
    for i, r in enumerate(sort_r):
        plt.figure()
        y_list = data[:,distance[r]]
        plt.plot(tc_list, y_list)
        plt.figtext(0.5, 0.98, max_E_label, ha="center", va="top", fontsize=10, color="blue")
        diff_max = f"{tc_list[np.argmax(np.diff(y_list))]:.3f}"
        plt.figtext(0.5, 0.93, f'r: {r}'+' change: '+diff_max, ha="center", va="top", fontsize=10, color="blue")
        plt.savefig(f'/content/current_Jr_maxE={max_E:.2f}_r={r}_{label}.png')
        plt.show()
        plt.close()
        if i > 5:
            break

def current_kwant(sys, num_states = 20, max_E = 0.5):
    H = sys.hamiltonian_submatrix(sparse=False)
    J = kwant.operator.Current(sys)
    sum_current = None
    
    if num_states is not None:
        evals, evecs = eigsh(H, k=num_states, sigma=0)
        max_E_label = 'max energy: '+ f'{max(evals, key=abs):.3f}'
        for i in range(num_states):
            current = J(evecs[:, i])
            if sum_current is None:
                sum_current = current
            else:
                sum_current += current
    else:
        evals, evecs = eigh(H)
        max_E_label = 'max energy: '+ f'{max_E:.3f}'
        for i, e in enumerate(evals):
            if abs(e) < max_E:
                current = J(evecs[:, i])
                if sum_current is None:
                    sum_current = current
                else:
                    sum_current += current
        
    fig, ax = plt.subplots()
    kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
    kwant.plotter.current(sys, sum_current, ax=ax, colorbar=True)
    label = pick_label(model['name'])
    plt.figtext(0.5, 0.98, label, ha="center", va="top", fontsize=10, color="blue")
    plt.figtext(0.5, 0.93, max_E_label, ha="center", va="top", fontsize=10, color="blue")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig(f'/content/current_kwant_maxE={max_E:.2f}_{label}.png')
    plt.show()
    plt.close()
    
# import pybinding as pb

def rectangle_vertex(L, W):
    cc = model['cc']
    return (L*cc/2, ((W-1)/2*1.5+1)*cc/sqrt(3))

def pick_label(name, iftc = True):
    if name == HALDANETRI:
        return name + '_' + model['category'] + '_' + f"t3={model['t3']:.2f}_tc={model['tc']:.2f}_t2={model['t2']:.2f}_L={model['L']}_W={model['W']}"
    if name == DEFECT:
        if iftc == False:
            return name + '_' + model['category'] + '_' + f"L={model['L']}_W={model['W']}"    
        return name + '_' + model['category'] + '_' + f"tc={model['tc']:.2f}_L={model['L']}_W={model['W']}"
    else:
        system.exit()

def haldane_triangular_pybinding():
#     cc = model['cc']
#     m, m2 = model['m'], model['m2']
#     L, W = model['L'], model['W']
#     t1, tc, t2, t3 = model['t1'], model['tc'], model['t2'], model['t3']
#     temp = tc*cmath.exp((1.j)*pi/2.)

#     def model_build():
#         lat = pb.Lattice(a1=[cc, 0],
#                         a2=[cc/2, cc* sqrt(3)/2 ])

#         lat.add_sublattices(('c', [0, 0], m2),
#                             ('a', [cc/2, -cc / (2*sqrt(3))], m),
#                             ('b', [cc/2, cc / (2*sqrt(3))], -m))

#         # lat.register_hopping_energies({
#         #     't1': kwargs.get('t1', t1),
#         #     'tc': kwargs.get('tc', tc),
#         #     't2c': kwargs.get('t2c', t2c),
#         # })

#         neighbors_second = [[0, 0], [-1, 1], [-1, 0]]
#         for neighbor in neighbors_second:
#             lat.add_hoppings((neighbor, 'c', 'a', -t3))
#         neighbors_second = [[0, 0], [0, -1], [-1, 0]]
#         for neighbor in neighbors_second:
#             lat.add_hoppings((neighbor, 'c', 'b', -t3))

#         neighbors_graphene = [[0, 0], [0, -1], [1, -1]]
#         for neighbor in neighbors_graphene:
#             lat.add_hoppings((neighbor, 'a', 'b', -t1))

#         neighbors_triangular = [(0, 1), (1, 0), (-1, 1)]
#         for neighbor in neighbors_triangular:
#             lat.add_hoppings((neighbor, 'c', 'c', -t2))

#         neighbors_imag = [(0, 1), (-1, 0), (1, -1)]
#         for neighbor in neighbors_imag:
#             lat.add_hoppings((neighbor, 'b', 'b', temp))
#             lat.add_hoppings((neighbor, 'a', 'a', temp.conjugate()))

#         return lat

#     def rectangle(L, W):
#         (x0, y0) = vertex(L, W)
#         x, y = x0+0.1, y0+0.1
#         return pb.Polygon([[x, y], [x, -y], [-x, -y], [-x, y]])

#     test = pb.Model(
#         model_build(),
#         # pb.rectangle(10),
#         # pb.translational_symmetry(a2=False),
#         rectangle(L, W),
#     )

#     return test
    return

def haldane_triangular():
    cc = model['cc']
    lat = kwant.lattice.general([(cc, 0), (cc / 2, cc * sqrt(3) / 2)],
                                 [(0, 0), (cc/2, -cc / (2*sqrt(3))), (cc/2, cc / (2 * sqrt(3)))],
                                 ['c', 'a', 'b'], norbs=1)
    c, a, b = lat.sublattices

    m, m2 = model['m'], model['m2']
    def onsite(site):
        if site.family.name == 'a':
            return m
        if site.family.name == 'b':
            return -m
        return m2

    L, W = model['L'], model['W']
    (x, y) = rectangle_vertex(L, W)
    if L == 0 and W == 0: # PEC: currently useless
        sys = kwant.Builder()
        sys[lat.shape((lambda pos: True), (0, 0))] = onsite
    elif L == 0: # ribbon: zigzag y
        sys = kwant.Builder(kwant.TranslationalSymmetry([cc, 0]))
        sys[lat.shape((lambda pos: -y<= pos[1] <=y), (0, 0))] = onsite
    else:
        sys = kwant.Builder()
        sys[lat.shape((lambda pos: (-x<=pos[0]<=x) and (-y<=pos[1]<= y)), (0, 0))] = onsite

    t1, tc, t2, t3 = model['t1'], model['tc'], model['t2'], model['t3']
    hoppings = [((0, 0), a, c), ((0, 0), b, c), ((-1, 1), a, c), ((0, -1), b, c), ((-1, 0), a, c), ((-1, 0), b, c)]
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t3
    neighbors_triangular = [(0, 1), (1, 0), (-1, 1)]
    for neighbor in neighbors_triangular:
        sys[kwant.builder.HoppingKind(neighbor, c, c)] = -t2
    neighbors_graphene = [(0, 0), (0, -1), (1, -1)]
    for neighbor in neighbors_graphene:
        sys[kwant.builder.HoppingKind(neighbor, b, a)] = -t1

    temp = tc*cmath.exp((1.j)*pi/2.)
    neighbors_imag = [(0, 1), (-1, 0), (1, -1)]
    for neighbor in neighbors_imag:
        sys[kwant.builder.HoppingKind(neighbor, b, b)] = temp
        sys[kwant.builder.HoppingKind(neighbor, a, a)] = temp.conjugate()

    kwant.plot(sys)

    # 2D PEC exception: currently useless
    if L == 0 and W == 0:
        return kwant.wraparound.wraparound(sys).finalized()
    return sys.finalized()

def spectral_localizer(sys, x, y, E):
    kappa = model['kappa']

    if model['name'] == PYBINDING:
        H = sys.hamiltonian.toarray(model)
        dim = np.shape(H)[0]
        X, Y = np.zeros(H.shape), np.zeros(H.shape)
        for i, site in enumerate(sys.system.positions.x):
            X[i, i] = site
        for i, site in enumerate(sys.system.positions.y):
            Y[i, i] = site
    else:
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
            eigvals = eigsh(L, k=num_eigvals, sigma=0, return_eigenvectors=False)
            min_eigvals[i][j] = np.min(np.abs(eigvals))

    plt.figure()
    plt.imshow(min_eigvals, cmap='viridis', interpolation='bicubic',
               extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]])
    plt.colorbar()
    plt.title('Spectral Localizer Minimum Eigenvalues')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    label = pick_label(model['name'])
    plt.figtext(0.5, 0.01, label, ha="center", va="bottom", fontsize=10, color="blue", bbox=dict(facecolor='lightblue', edgecolor='blue'))
    plt.savefig(f'/content/localizer_gap_{label}.png')
    plt.show()
    plt.close()

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

def sigma_change():# precondition: eigenvalues_change already cross 0
    change_model(DEFECT, SINGLE)
    sigma_value = []
    tc_value = np.linspace(0.8, 1.3, 100)
    for tc in tc_value:
        model['tc'] = tc
        sys = model_builder()
        L, dim = spectral_localizer(sys, 0, 0, 0)
        # only apply to haldane_defect because the smallest one may not be on the band crossing 0
        eigenvalue = eigsh(L, k=1, sigma=0, return_eigenvectors=False, tol=1e-5)
        sigma_value.append(eigenvalue[0])
        
    plt.plot(tc_value, sigma_value)
    plt.axhline(0, color='red', linewidth=1.5, linestyle='--')
    plt.ylabel('sigma(origin) change')
    label = pick_label(model['name'])
    plt.figtext(0.5, 0.01, label, ha="center", va="bottom", fontsize=10, color="blue")
    
    f = interp1d(tc_value, sigma_value)
    zero_1 = fsolve(f, x0=0.9)
    zero_2 = fsolve(f, x0=1.2)

    str_zero_point = f'zero crossing:{zero_1[0]:.3f}, {zero_2[0]:.3f}'
    plt.figtext(0.5, 0.8, str_zero_point, ha="center", va="top", fontsize=10, color="blue")
    plt.savefig(f'/content/sigma_change_{label}.png')
    plt.show()
    plt.close()
    
def eigenvalues_change(sys):
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
        L, dim = spectral_localizer(sys, y, coord_fix, 0)
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
    for i in range(num_eigvals):
        line = tracked_eigvals[:, i]
        if np.max(np.abs(np.diff(line))) > CONTINUEPOINT:
            continue

        signs = np.sign(line)
        changes = np.diff(signs)
        crossing_indices = np.where(changes != 0)[0]

        for index in crossing_indices:
            #print(crossing_indices, x_coords[index], line[index], x_coords[index+1], line[index+1])
            zero_point.append((v_coords[index]+v_coords[index+1])/2)

        plt.plot(v_coords, line, label=f'Eig {i+1}')
    plt.ylabel('localizer eigenvalues')
    label = pick_label(model['name'])
    plt.figtext(0.5, 0.01, label, ha="center", va="bottom", fontsize=10, color="blue")

    str_zero_point = 'zero point: ' + ', '.join([f'{item:.3f}' for item in zero_point])
    plt.figtext(0.5, 0.96, str_zero_point, ha="center", va="top", fontsize=10, color="blue")
    plt.figtext(0.5, 0.92, xlabel, ha="center", va="top", fontsize=10, color="blue")
    plt.savefig(f'/content/eigenvalues_change_{label}_{xlabel}.png')
    plt.show()
    plt.close()

def change_model(name, category):
    if name==HALDANETRI or name==PYBINDING:
        if category==NONTRIVIAL:
            model.update(dict(name = name, category = category,
                        cc = 1.0,
                        m = 0.0, t1 = 1.0, tc = 0.5,
                        t2 = 0.2, m2 = -0.35,
                        t3 = 0.3,
                        kappa = 1.0,
                        L=11, W=11))
        elif category==TRIVIAL:
            model.update(dict(name = name, category = category,
                        cc = 1.0,
                        m = 2*sqrt(3), t1 = 1.0, tc = 0.5,
                        t2 = 0.2, m2 = -0.35,
                        t3 = 0.3,
                        kappa = 1.0,
                        L=11, W=11))
    elif name==DEFECT:
        model.update(dict(name=name, category = category,
                      cc = 1.0,
                      m = 0.0, t1 = 1.0, tc = 1.0,
                      kappa = 1.0,
                      L=13, W=13))
    else:
        print('change_model error')
        system.exit()

def change_para(func):
    L, W, cc = model['L'], model['W'], model['cc']
    (x, y) = rectangle_vertex(L, W)
    print('x=', x, ', y=', y)

    if func==GAP:
        if model['name'] == HALDANETRI: # 1/4 pic
            para.update(dict(func = func,
                             y_min = -y-cc, y_max = 0,
                             x_min = -x-cc, x_max = 0,
                             num_cc = 5, num_eigvals = 3))
        elif model['name'] == DEFECT: # whole pic
            para.update(dict(func = func,
                             y_min = -y, y_max = y,
                             x_min = -x, x_max = x,
                             num_cc = 5, num_eigvals = 10))
    elif func==CHANGE:
        if model['name'] == HALDANETRI: # x_fix = 0 or cc/2
            para.update(dict(func = func,
                             y_min = -y-cc, y_max = -y/2, num_cc=200,
                             x_min = 0, x_max = 0,
                             num_eigvals = 20,))
        elif model['name'] == DEFECT: # x_fix = 0
            # y_min = -const*cc, y_max = const*cc
            para.update(dict(func = func,
                             y_min = -3*cc, y_max = 3*cc, num_cc=200,
                             x_min = 0, x_max = 0,
                             num_eigvals = 6,))
    else:
        print('change_para error')
        system.exit()

def main_func(name, category):
    change_model(name, category)
    sys = model_builder()

    # change_para(GAP)
    # localizer_gap(sys)

    change_para(CHANGE)
    eigenvalues_change(sys)

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
            for tc in [0.1, 3]:
                  model['t3'], model['tc'], model['t2'] = t3, tc, t2
                  sys = model_builder()

                  change_para(GAP)
                  localizer_gap(sys)

                  # change_para(CHANGE)
                  # eigenvalues_change(sys)

import os
import shutil
def sync_png_files():
    source_dir = '/content'
    target_dir = '/content/drive/My Drive/Colab Notebooks/Synced Images'
    os.makedirs(target_dir, exist_ok=True)
    for file_name in os.listdir(source_dir):
            if file_name.endswith('.png'):
                source_path = os.path.join(source_dir, file_name)
                destination_path = os.path.join(target_dir, file_name)

                shutil.copy(source_path, destination_path)
# band_structure()
#different_haldane()
# sync_png_files()

current_Jr(DEFECT, SINGLE, max_E=0.2)