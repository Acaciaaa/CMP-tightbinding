from __future__ import division
import kwant
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys as system
import cmath
from math import sqrt, pi

'''Defining Pauli matrices'''
s0 = np.array([[1,0],[0,1]], complex)
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]], complex)
SSH, HALDANE, HALDANETRI, DEFECT, PYBINDING = 'SSH', 'haldane', 'haldane and triangular', 'defect', 'pybinding'
NONTRIVIAL, TRIVIAL, NOMASS, SINGLE, CLUSTER, NONE = 'nontrivial', 'trivial', 'nomass', 'single', '7 hexagons', 'none'
model = dict(name = NONE,  category = NONE,
         cc = 1.0, # cell-cell distance not site-site distance
         m = 1.0, t1 = 1/2, h = 1/2, # on-site difference -> 2m, 1st neighbor -> t1, 2nd neighbor -> t2
         t2 = 1/2, m2 = 1/2, # nearest neighbor -> t2, on-site energy -> m2
         t3 = 1.0, # coupling between two lattices
         kappa = 1.0, deltaE = 1.0,
         L=4, W=4)
GAP, CHANGE = 'localizer gap', 'minimum eigenvalues change'
para = dict(func = NONE,
            y_min = 0, y_max = model['W'],
            x_min = 0, x_max = model['L'],
            num_cc = 50, num_eigvals = 10)
def rectangle_vertex(L, W):
    cc = model['cc']
    return (L*cc/2, ((W-1)/2*1.5+1)*cc/sqrt(3))
def model_builder():
    if model['name'] == DEFECT:
        return defect_graphene()
    if model['name'] == HALDANETRI:
        return haldane_triangular()
    if model['name'] == HALDANE:
        return haldane()
    if model['name'] == SSH:
        return ssh()
    # return haldane_triangular_pybinding()

def ssh():
    lat = kwant.lattice.chain(norbs=1)
    sys = kwant.Builder()
    L, t1, t2 = model['L'], model['h'], model['t2']
    for i in range(2*L):
        sys[lat(i)] = 0

    for i in range(2*L-1):
        if i % 2 == 0: # A->B weak
            sys[lat(i), lat(i + 1)] = -t1
        else: # B->A strong
            sys[lat(i), lat(i + 1)] = -t2
    
    sys = sys.finalized()
    #kwant.plot(sys)
    def draw_model():
        fig, ax = plt.subplots()
        kwant.plot(sys, ax=ax,site_size=0.2)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_frame_on(False)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        for site in sys.sites:
            x = site.pos[0]
            y=-0.5
            ax.text(x, y, f'{x:.0f}', fontsize=12, ha='right', va='bottom', color='black')
            if x%2 == 0:
                ax.text(x, 0.2, 'A', fontsize=12, ha='right', va='bottom', color='black')
            else:
                ax.text(x, 0.2, 'B', fontsize=12, ha='right', va='bottom', color='black')
        #plt.show()
        fig.savefig(f"/Users/ruiqixu/Desktop/ssh_model.png", dpi=300, bbox_inches='tight', pad_inches=0)
    #draw_model()
    return sys

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

    t1, h, = model['t1'], model['h']

    neighbors_graphene = [(0, 0), (0, -1), (1, -1)]
    for neighbor in neighbors_graphene:
        sys[kwant.builder.HoppingKind(neighbor, b, a)] = -t1

    temp = h*cmath.exp((1.j)*pi/2.)
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
    
    def draw_model():
        def color_sites(site):
            if 'a' in site.family.name or 'b' in site.family.name:
                return mcolors.to_rgba("black", alpha=0.8)
        def color_hoppings(tar, sour):
            if sour.family != tar.family:  
                return 'black' 
            else:
                return 'dimgray'
        
        fig, ax = plt.subplots()
        kwant.plot(sys, ax=ax,site_color=color_sites,site_size=0.08,hop_color=color_hoppings,hop_lw=0.05)
        ax.set_aspect('equal', 'box')
        ax.plot([0, 2.6], [0, 0], color='crimson', linewidth=2, linestyle='--')
        hex_pos = np.array([[0.5, -0.5/sqrt(3)], [0.5, 0.5/sqrt(3)], [0, 1/sqrt(3)], [-0.5, 0.5/sqrt(3)], [-0.5, -0.5/sqrt(3)], [0, -1/sqrt(3)]])
        arrow_pos, arrow_k, x_k = [], [], [-1,-1,2,-2,1,1]
        for i in [0, 2, 4, 1, 3, 5]:
            j = (i+2)%6
            arrow_pos.append((hex_pos[i]+hex_pos[j])/2)
            arrow_k.append((hex_pos[j][1]-hex_pos[i][1])/(hex_pos[j][0]-hex_pos[i][0]))
        epsilon, ratio = 0.1, 0.5
        for i in range(6):
            x, y, delta_x = arrow_pos[i][0], arrow_pos[i][1], epsilon*x_k[i]
            ax.annotate('', xy=(x+ratio*delta_x, y+ratio*delta_x*arrow_k[i]), xytext=(x-(1-ratio)*delta_x, y-(1-ratio)*delta_x*arrow_k[i]),
                        arrowprops=dict(arrowstyle='-|>', color='dimgray', lw=0.1,mutation_scale=16))
        ax.axis('off')
        #plt.tight_layout()
        plt.savefig(f"/Users/ruiqixu/Desktop/model.png",dpi=fig.dpi, bbox_inches='tight')
        #plt.show()
    
    #draw_model()
    #kwant.plot(sys)
    
    return sys.finalized()

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

    t1, h, t2, t3 = model['t1'], model['h'], model['t2'], model['t3']
    hoppings = [((0, 0), a, c), ((0, 0), b, c), ((-1, 1), a, c), ((0, -1), b, c), ((-1, 0), a, c), ((-1, 0), b, c)]
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t3
    neighbors_triangular = [(0, 1), (1, 0), (-1, 1)]
    for neighbor in neighbors_triangular:
        sys[kwant.builder.HoppingKind(neighbor, c, c)] = -t2
    neighbors_graphene = [(0, 0), (0, -1), (1, -1)]
    for neighbor in neighbors_graphene:
        sys[kwant.builder.HoppingKind(neighbor, b, a)] = -t1

    temp = h*cmath.exp((1.j)*pi/2.)
    neighbors_imag = [(0, 1), (-1, 0), (1, -1)]
    for neighbor in neighbors_imag:
        sys[kwant.builder.HoppingKind(neighbor, b, b)] = temp
        sys[kwant.builder.HoppingKind(neighbor, a, a)] = temp.conjugate()

    kwant.plot(sys)

    # 2D PEC exception: currently useless
    if L == 0 and W == 0:
        return kwant.wraparound.wraparound(sys).finalized()
    return sys.finalized()

def haldane():
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

    t1, h = model['t1'], model['h']
    neighbors_graphene = [(0, 0), (0, -1), (1, -1)]
    for neighbor in neighbors_graphene:
        sys[kwant.builder.HoppingKind(neighbor, b, a)] = -t1

    temp = h*cmath.exp((1.j)*pi/2.)
    neighbors_imag = [(0, 1), (-1, 0), (1, -1)]
    for neighbor in neighbors_imag:
        sys[kwant.builder.HoppingKind(neighbor, b, b)] = temp
        sys[kwant.builder.HoppingKind(neighbor, a, a)] = temp.conjugate()

    #kwant.plot(sys)
    
    return sys.finalized()

def change_model(name, category):
    if name==HALDANETRI or name==PYBINDING:
        if category==NONTRIVIAL:
            model.update(dict(name = name, category = category,
                        cc = 1.0,
                        m = 0.0, t1 = 1.0, h = 0.5,
                        t2 = 0.2, m2 = -0.35,
                        t3 = 0.3,
                        kappa = 1.0,
                        L=11, W=11))
        elif category==TRIVIAL:
            model.update(dict(name = name, category = category,
                        cc = 1.0,
                        m = 2*sqrt(3), t1 = 1.0, h = 0.5,
                        t2 = 0.2, m2 = -0.35,
                        t3 = 0.3,
                        kappa = 1.0,
                        L=11, W=11))
    elif name==DEFECT:
        model.update(dict(name=name, category = category,
                      cc = 1.0,
                      m = 0.0, t1 = 1.0, h = 1,
                      kappa = 1.0,
                      L=13, W=13))
    elif name==HALDANE:
        model.update(dict(name = name, category = category,
                        cc = 1.0,
                        m = 2*sqrt(3), t1 = 1.0, h = 0,
                        kappa = 1.0,
                        L=7, W=7))
        if category==NOMASS:
            model['m'] = 0
        if category==NONTRIVIAL:
            model['h'] = 1
        elif category==TRIVIAL:
            model['h'] = 0.5
    elif name == SSH:
        model.update(dict(name=name, category=category,
                          cc=1.0,
                          h=0.5, t2=1,
                          L=6))
    else:
        print('change_model error')
        system.exit()

def change_para(func):
    L, W, cc = model['L'], model['W'], model['cc']
    (x, y) = rectangle_vertex(L, W)
    print('x=', x, ', y=', y)

    if func==GAP:
        if model['name'] == HALDANETRI or model['name'] == HALDANE: # 1/4 pic
            para.update(dict(func = func,
                             y_min = -y, y_max = 0,
                             x_min = -x, x_max = 0,
                             num_cc = 5, num_eigvals = 3))
        elif model['name'] == DEFECT: # 1/4 pic
            para.update(dict(func = func,
                             y_min = -y, y_max = 0,
                             x_min = -x, x_max = 0,
                             num_cc = 5, num_eigvals = 2))
    elif func==CHANGE:
        if model['name'] == HALDANETRI or model['name'] == HALDANE: # x_fix = 0 or cc/2
            para.update(dict(func = func,
                             y_min = -y, y_max = -y/2, num_cc=200,
                             x_min = 0, x_max = 0,
                             num_eigvals = 20,))
        elif model['name'] == DEFECT: # x_fix = 0
            # y_min = -const*cc, y_max = const*cc
            para.update(dict(func = func,
                             y_min = 0, y_max = 0, num_cc=200,
                             x_min = -x, x_max = 0,
                             num_eigvals = 20,))
        elif model['name'] == SSH:
            para.update(dict(func = func,
                             y_min = 0, y_max = 0, num_cc=200,
                             x_min = -0.5, x_max = 2,#2*model['L']-0.5,
                             num_eigvals = 2,))
    else:
        print('change_para error')
        system.exit()
    return x, y