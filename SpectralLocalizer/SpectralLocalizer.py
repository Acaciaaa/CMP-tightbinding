from __future__ import division
import kwant
import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from numpy.linalg import eigh
from scipy.linalg import kron
# import pylab as py
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

HALDANE, HALDANETRI, DEFECT, PYBINDING = 'haldane', 'haldane and triangular', 'defect graphene', 'pybinding'
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

def model_builder():
    if model['name'] == DEFECT:
        return defect_graphene()
    if model['name'] == HALDANETRI:
        return haldane_triangular()
    if model['name'] == HALDANE:
        return haldane()
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
        
    #kwant.plot(sys)
    return sys.finalized()

from scipy.stats import gaussian_kde
import matplotlib.patches as patches
def custom_sort(evals, evecs, ifcurrent=False):
    indices = np.argsort(np.abs(evals))
    sorted_evals = evals[indices]
    if ifcurrent:
        sorted_eigens = evecs[indices, :]
    else:
        sorted_eigens = evecs[:, indices]
    return sorted_evals, sorted_eigens
def current_Jr(name, category):
    CUTOFF, GAUSSIAN, STATE = 'max_E', 'gaussian', 'state'

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
    
    def find_distribution(evals):
        plt.hist(evals, bins=50, range = (-2, 2), color='blue', alpha=0.7)
        tick_marks = np.linspace(-2, 2, 51) 
        plt.xticks(tick_marks, rotation=90, fontsize=7) 
        plt.title('Energy Frequency Distribution: h=' + f"{model['h']}")
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.savefig(f'/content/energy_distribution_{label}.png')
        plt.grid(True)
        plt.show()
        plt.close()
        
        closest_to_zero = sorted(evals, key=abs)[:50]
        print(closest_to_zero)
    
    def draw_distribution():
        h_num, energy_num = 50, 6
        h_list = np.linspace(0.3, 1.5, h_num)
        energy_list = np.zeros((energy_num, h_num))
        for i, h in enumerate(h_list):
            model['h'] = h
            sys = model_builder()
            H = sys.hamiltonian_submatrix(sparse=False)
            evals, _ = eigh(H)
            evals = np.abs(sorted(evals, key=abs))
            energy_list[:, i] = evals[0:energy_num*2:2]
        plt.figure()
        plt.xlabel('h')
        plt.ylabel('energy')
        # plt.ylim(0, 0.01)
        for i in range(energy_num):
            plt.plot(h_list, energy_list[i], label=f'{i}')
        plt.legend()
        plt.show()
        plt.close()
    
    def pure_current_info(sys):
        H = sys.hamiltonian_submatrix(sparse=False)
        J = kwant.operator.Current(sys)
        evals, evecs = eigh(H)
        current = np.array([J(evecs[:, i]) for i in range(len(evals))])
        #print(np.shape(evals), np.shape(evecs), np.shape(current))
        return evals, current
    
    def current_filter(evals, current, tag, *args):
        sum_current = None
        
        if tag == CUTOFF:
            max_E = args[0]
            way = f"cutoff_max_E={max_E:.2f}"
            for i, e in enumerate(evals):
                if e > 0 and e <= max_E:
                    if sum_current is None:
                        sum_current = current[i]
                    else:
                        sum_current += current[i]
            
        elif tag == GAUSSIAN:
            max_E, E0, sigma = args[0], args[1], args[2]
            way = f"gaussian_maxE={max_E:.2f}_E0={E0:.2f}_sigma={sigma:.2f}"
            gaussian_values = np.exp(-(evals - E0)**2 / (2 * sigma**2))
            gaussian_values /= np.max(gaussian_values)
            gaussian_values[evals <= E0] = 1
            gaussian_values[evals >= max_E] = 0
            gaussian_values[evals < 0] = 0

            for i, e in enumerate(gaussian_values):
                if e > 0:
                    if sum_current is None:
                        sum_current = e * current[i]
                    else:
                        sum_current += e * current[i]

        elif tag == STATE:
            start, stop = args[0], args[1]
            way = f"state_start={start}_stop={stop}"
            sorted_evals, sorted_current = custom_sort(evals, current, True)
            sum_current = sorted_current[2*start]#+J(sorted_evecs[:, 2*start+1])
            for i in range(start+1, stop):
                sum_current += sorted_current[2*i]#+J(sorted_evecs[:, 2*i+1])
            
        return way, sum_current
    
    def draw_current():
        h_list = [0.0, 0.8]#, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
        bounds = [(-7, 7), (-6, 6)]
        #start, stop = 0, 1
        for start in range(10):
            stop = start+1
            for h in h_list:
                model['h'] = h
                sys = model_builder()
                fig, ax = plt.subplots()
                ax.set_aspect('equal')
                plt.xlim(bounds[0][0], bounds[0][1])
                plt.ylim(bounds[1][0], bounds[1][1])
                #signature_change3(sys, e=0, bounds=bounds)
                evals, current = pure_current_info(sys)
                way, sum_current = current_filter(evals, current, STATE, start, stop)
                index = -1
                for head, tail in sys.graph:
                    index += 1
                    if abs(sum_current[index]) < 1e-10:
                        continue
                    start_point, end_point = sys.sites[head].pos, sys.sites[tail].pos
                    #x_start, y_start, x_end, y_end = round(p1[0], 3), round(p1[1], 3), round(p2[0], 3), round(p2[1], 3)
                    if np.linalg.norm(start_point) > 20:
                        continue
                    weight = sum_current[index]
                    if weight < 0:
                        start_point, end_point = end_point, start_point
                        weight = -weight
                    #mid_x = (x_start + x_end) / 2
                    #mid_y = (y_start + y_end) / 2
                    #r = sqrt(mid_x*mid_x+mid_y*mid_y)
                    #if r > 4:
                    #    continue
                    normalized = (end_point-start_point)/np.linalg.norm(end_point - start_point)
                    arrow_length = normalized * weight * 100

                    

                    arrow = patches.FancyArrowPatch(start_point, start_point+arrow_length,
                                                    arrowstyle='-|>', connectionstyle='arc3,rad=0.0', mutation_scale=10, color='blue')
                    ax.add_patch(arrow)
                    # if 0<=mid_x<=3 and 0<=mid_y<=3:
                    #     ax.text(mid_x, mid_y, f'{weight:.4f}'.lstrip('0').replace('-0.', '-.'), color='red', fontsize=8, ha='center', va='center')
                kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
                plt.title(f"h={h}")
                plt.savefig(f"\content\Chern_Current_h={h:.2f}.png")
                plt.show()
                plt.close()

    def magnitude_info(sum_current):
        temp = [None] * len(sort_r)
        magnitude = np.zeros(len(sort_r))
        sum_direct, sum_angular = 0, 0
        for index, current in enumerate(sum_current):
            result = current * pos_info[index][0]
            r = pos_info[index][1]
            if temp[sort_distance[r]] is None:
                temp[sort_distance[r]] = [result]
            else:
                temp[sort_distance[r]].append(result)
        
        # not for drawing J(r)
        # before = [None] * len(sort_r)
        # for index, current in enumerate(sum_current):
        #     r = pos_info[index][1]
        #     if before[sort_distance[r]] is None:
        #         before[sort_distance[r]] = [current]
        #     else:
        #         before[sort_distance[r]].append(current)
                
        # print(model['tc'])        
        # for r in sort_r:
        #     if sort_distance[r] < 5:
        #         print(r, sort_distance[r])
        #         print('before: ', before[sort_distance[r]])
                
        for key, value in sort_distance.items():
            # if value == 3:
            #     print(value, ': ',temp[value])
            tmp = np.sum(temp[value])
            if value >= 0:
                sum_direct += tmp
                sum_angular += tmp*key
            magnitude[value] = tmp/len(temp[value])
            # print(len(temp[value]), key)
        return magnitude, sum_direct, sum_angular
    
    change_model(name, category)
    def pos():
        temp_sys = model_builder()
        distance, pos_info = edge_info(temp_sys)
        sort_r = sorted(distance.keys())
        sort_distance = {key: index for index, key in enumerate(sort_r)}
        return pos_info, sort_r, sort_distance
    pos_info, sort_r, sort_distance = pos()
    # draw_distribution()
    # draw_current()
    
    def draw_h_fixed(tag, *args):
        h_list = [0.0, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 5.0]
        start, stop = 0, 6
        for h in h_list:
            model['h'] = h
            sys = model_builder()
            label = pick_label(model['name'])
            evals, current = pure_current_info(sys)
            way, sum_current = current_filter(evals, current, tag, *args)
            # draw_current()
            magnitude, _, _ = magnitude_info(sum_current)
            multiply_item = magnitude# * sort_r
            
            plt.figure()
            plt.scatter(sort_r, multiply_item, s=1)
            plt.plot(sort_r, multiply_item, marker='o')
            plt.axhline(0, color='grey', linewidth=1)
            plt.ylim(-0.18, 0.1)
            plt.xlabel('distance r')
            plt.ylabel('J(r)')
            plt.figtext(0.5, 0.95, f"h={h:.2f} {way}", ha="center", va="top", fontsize=10, color="blue")
            plt.savefig(f'/content/current_Jr_{way}_{label}.png')
            plt.show()
            plt.close()
    # draw_h_fixed(GAUSSIAN, 0.2, 0.0, 0.1)

    def draw_r_fixed(r):
        h_list = np.linspace(0.3, 1.5, 50)
        Jr_list = []
        for h in h_list:
            model['h'] = h
            sys = model_builder()
            label = pick_label(model['name'],ifh=False)
            sum_current = current_info(sys, find_energy_distribution=False)
            magnitude, _, _ = magnitude_info(sum_current)
            Jr_list.append(magnitude[sort_distance[r]])

        plt.figure()
        plt.scatter(h_list, Jr_list, s=1)
        plt.plot(h_list, Jr_list, marker='o')
        plt.axhline(0, color='grey', linewidth=1)
        #plt.ylim(-0.14, 0.08)
        plt.xlabel('h')
        plt.ylabel('average current')
        plt.figtext(0.5, 0.93, max_E_label+f' r: {r:.3f}', ha="center", va="top", fontsize=10, color="blue")
        plt.figtext(0.5, 0.97, label, ha="center", va="top", fontsize=10, color="blue")
        plt.savefig(f'/content/Jr_r={r:.3f}_maxE={max_E:.2f}_{label}.png')
        plt.show()
        plt.close()
    # draw_r_fixed(sort_r[4])

    # always change due to diff requirements
    def draw_sum(whichsum, tag, *args):
        h_num = 50
        part1 = np.linspace(0.3, 0.9, 60)
        part2 = np.linspace(0.9, 1.1, 100)
        part3 = np.linspace(1.1, 1.5, 40)
        h_list = np.concatenate((part1, part2, part3))
        #h_list = np.concatenate((np.linspace(0.3, 1.3, 100), np.linspace(1.3, 9.3, 16))) large h
        h_list = np.linspace(0.3, 2, h_num)
        
        def diff_size():
            nonlocal pos_info, sort_r, sort_distance
            plt.figure()
            for L in [9, 13, 17]:
                model['L'] = L
                pos_info, sort_r, sort_distance = pos()
                sum_list = []
                for i, h in enumerate(h_list):
                    model['h'] = h
                    sys = model_builder()
                    evals, current = pure_current_info(sys)
                    way, sum_current = current_filter(evals, current, tag, *args)
                    if whichsum == 'J(r)':
                        _, sum, _ = magnitude_info(sum_current)
                    elif whichsum == 'J(r)_r':    
                        _, _, sum = magnitude_info(sum_current)
                    # if sum > 0 and sum_list[i-1] < 0:
                    #     zero_points = [h_list[i-1], h]
                    sum_list.append(sum)
                plt.plot(h_list, sum_list, label=f'L={L}')
            plt.axhline(0, color='grey', linewidth=1)
            plt.xlabel('h')
            plt.legend()
            plt.ylabel('sum of '+whichsum)
            # plt.figtext(0.5, 0.92, f"cross:[{zero_points[0]:.3f}, {zero_points[1]:.3f}]", ha="center", va="top", fontsize=10, color="blue")
            plt.figtext(0.5, 0.97, way, ha="center", va="top", fontsize=10, color="blue")
            # plt.savefig(f'/content/sum_{whichsum}_{way}_{label}.png')
            plt.show()
            plt.close()
        
        def diff_state():
            state_num = 6
            sum_list = np.zeros((state_num, h_num))
            for i, h in enumerate(h_list):
                model['h'] = h
                sys = model_builder()
                evals, current = pure_current_info(sys)
                for n in range(state_num):
                    way, sum_current = current_filter(evals, current, tag, n, n+1)
                    if whichsum == 'J(r)':
                        _, sum, _ = magnitude_info(sum_current)
                    elif whichsum == 'J(r)_r':    
                        _, _, sum = magnitude_info(sum_current)
                    sum_list[n, i] = sum
            plt.figure()
            for n in range(state_num):
                plt.scatter(h_list, n, label=f"{n}")
            plt.axhline(0, color='grey', linewidth=1)
            plt.xlabel('h')
            plt.legend()
            plt.ylabel('sum of '+whichsum)
            # plt.figtext(0.5, 0.92, f"cross:[{zero_points[0]:.3f}, {zero_points[1]:.3f}]", ha="center", va="top", fontsize=10, color="blue")
            plt.figtext(0.5, 0.97, way, ha="center", va="top", fontsize=10, color="blue")
            # plt.savefig(f'/content/sum_{whichsum}_{way}_{label}.png')
            plt.show()
            plt.close()

        diff_state()
    draw_sum('J(r)', STATE)# -> diff_state
    # draw_sum('J(r)', GAUSSIAN, 0.2, 0, 0.05) -> diff_size
    # draw_sum('J(r)', GAUSSIAN, 0.2, 0, 0.1) -> diff_size
    
    # diff L and sigma
    def zero_points_Gaussian():
        nonlocal pos_info, sort_r, sort_distance
        whichsum = 'J(r)'
        max_E = 0.2
        E0 = 0.0
        sigma_num = 70
        sigma_list = np.linspace(0.03, 0.1, sigma_num)
        h_list = np.linspace(0.9, 1.2, 300)
        def magnitude_filter():
            if whichsum == 'J(r)':
                _, sum, _ = magnitude_info(sum_current)
            elif whichsum == 'J(r)_r':    
                _, _, sum = magnitude_info(sum_current)
            return sum
        
        for L in [9, 10, 11, 12, 13, 14, 15, 16, 17]:
            model['L'] = L
            pos_info, sort_r, sort_distance = pos()
            storage = []
            for i, h in enumerate(h_list):
                model['h'] = h
                sys = model_builder()
                evals, current = pure_current_info(sys)
                tmp = []
                for sigma in sigma_list:
                    _, sum_current = current_filter(evals, current, GAUSSIAN, max_E, E0, sigma)
                    tmp.append(magnitude_filter())
                storage.append(tmp)
            storage = np.array(storage)
            change_points = np.zeros(sigma_num)
            for i in range(sigma_num):
                change_index = np.argmax(storage[:, i] >= 0)
                change_points[i] = (h_list[change_index-1] + h_list[change_index])/2
            plt.plot(sigma_list, change_points, label = f"L={L}")

        plt.xlabel('sigma')
        plt.ylabel('h')
        plt.title(whichsum)
        plt.legend()
        plt.show()
    # zero_points_Gaussian()

    def draw_circle_1():
        h_list = np.linspace(0.3, 1.5, 50)
        start, stop = 1, 2
        # for n in range(start, stop):
        inner_loop, outer_loop1, outer_loop2, outer_loop3 = [], [], [], []
        for h in h_list:
            model['h'] = h
            sys = model_builder()
            # sum_current = current_info(sys, find_energy_distribution=False)
            sum_current = current_info_onestate(sys, start, stop)
            magnitude, _, _ = magnitude_info(sum_current)
            inner_loop.append(magnitude[0])
            outer_loop1.append(magnitude[1])
            outer_loop2.append((magnitude[3]*12+magnitude[4]*6)/18)
            outer_loop3.append((magnitude[6]*12+magnitude[7]*12+magnitude[8]*6)/30)
        
        
        plt.figure()
        plt.axhline(0, color='grey', linewidth=1)
        for i, loop in enumerate([inner_loop, outer_loop1, outer_loop2, outer_loop3]):
            plt.scatter(h_list, loop, s=1)
            plt.plot(h_list, loop, marker='o')
            
            #plt.ylim(-0.14, 0.08)
            # diffs = np.abs(np.diff(loop))
            # max_diff = np.max(diffs)
            # second_max_diff = np.max(diffs[diffs != max_diff])
            # index_list = np.sort([np.where(diffs == max_diff)[0][0], np.where(diffs == second_max_diff)[0][0]])
            # plt.xlabel('h')
            # plt.ylabel('loop')
            # plt.figtext(0.5, 0.93, max_E_label+f" max diff: {(h_list[index_list[0]]+h_list[index_list[0]+1])/2:.3f}, {(h_list[index_list[1]]+h_list[index_list[1]+1])/2:.3f}", ha="center", va="top", fontsize=10, color="blue")
            # plt.figtext(0.5, 0.97, f'loop: {i}', ha="center", va="top", fontsize=10, color="blue")
            # if key_word == 'J(r)':
            #     plt.savefig(f'/content/sumdirect_maxE={max_E:.2f}_{label}.png')
            # else:
            #     plt.savefig(f'/content/sumangular_maxE={max_E:.2f}_{label}.png')
        plt.show()
        plt.close()
    # draw_circle_1()

    def draw_circle_2():
        h_list = np.linspace(0.6, 3, 100)
        start, stop = 180, 189
        loop = [[], [], [], []]
        for i, n in enumerate(range(start, stop)):
            for circle in range(4):
                loop[circle].append([])
            for h in h_list:
                model['h'] = h
                sys = model_builder()
                # sum_current = current_info(sys, find_energy_distribution=False)
                sum_current = current_info_onestate(sys, n, n+1)
                magnitude, _, _ = magnitude_info(sum_current)
                loop[0][i].append(magnitude[0])
                loop[1][i].append(magnitude[1])
                loop[2][i].append((magnitude[3]*12+magnitude[4]*6)/18)
                loop[3][i].append((magnitude[6]*12+magnitude[7]*12+magnitude[8]*6)/30)
        
        for circle in range(4):
            plt.figure()
            plt.axhline(0, color='grey', linewidth=1)
            for i, n in enumerate(range(start, stop)):
                #plt.scatter(h_list, loop[circle][n], s=1)
                plt.plot(h_list, loop[circle][i], label=f'{n}')
            
            #plt.ylim(-0.14, 0.08)
            # diffs = np.abs(np.diff(loop))
            # max_diff = np.max(diffs)
            # second_max_diff = np.max(diffs[diffs != max_diff])
            # index_list = np.sort([np.where(diffs == max_diff)[0][0], np.where(diffs == second_max_diff)[0][0]])
            # plt.xlabel('h')
            # plt.ylabel('loop')
            # plt.figtext(0.5, 0.93, max_E_label+f" max diff: {(h_list[index_list[0]]+h_list[index_list[0]+1])/2:.3f}, {(h_list[index_list[1]]+h_list[index_list[1]+1])/2:.3f}", ha="center", va="top", fontsize=10, color="blue")
            # plt.figtext(0.5, 0.97, f'loop: {i}', ha="center", va="top", fontsize=10, color="blue")
            # if key_word == 'J(r)':
            #     plt.savefig(f'/content/sumdirect_maxE={max_E:.2f}_{label}.png')
            # else:
            #     plt.savefig(f'/content/sumangular_maxE={max_E:.2f}_{label}.png')
            plt.legend()
            plt.show()
            plt.close()
    #draw_circle_2()

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
    
def current_direction(name, category, max_E = 0.1):
    change_model(name, category)
    temp_sys = model_builder()
    pos_info = []
    for head, tail in temp_sys.graph:
        p1, p2 = temp_sys.sites[head].pos, temp_sys.sites[tail].pos
        # unit_vector = (p2-p1)/np.linalg.norm(p2-p1)
        pos_info.append([p1, p2])
    
    last_current = None
    num_edges = []
    h_list = [0.6, 0.96, 1.2, 3]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for h_index, h in enumerate(h_list):
        model['h'] = h
        sys = model_builder()
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
        mask = np.isclose(sum_current, 0)
        sum_current[mask] = 0
        if last_current is not None:
            signs_differ = np.sign(sum_current) != np.sign(last_current)
            index = np.where(signs_differ)[0]
            for i in index:
                # print(pos_info[i][0], pos_info[i][1])
                color_map = {1:'purple',2:'orange',3:'red'}
                ax.plot([pos_info[i][0][0], pos_info[i][1][0]], [pos_info[i][0][1], pos_info[i][1][1]], color=color_map[h_index])
        last_current = sum_current[:]
    
    kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
    # plt.title(f"{tc_list[tc_index-1]}-{tc}")
    plt.show()
    plt.close()

# import pybinding as pb

def rectangle_vertex(L, W):
    cc = model['cc']
    return (L*cc/2, ((W-1)/2*1.5+1)*cc/sqrt(3))

def pick_label(name, ifh = True):
    if name == HALDANETRI:
        return name + '_' + model['category'] + '_' + f"t3={model['t3']:.2f}_h={model['h']:.2f}_t2={model['t2']:.2f}_L={model['L']}_W={model['W']}"
    if name == DEFECT or name == HALDANE:
        if ifh == False:
            return name + '_' + model['category'] + f"_kappa={model['kappa']}_L={model['L']}_W={model['W']}"    
        return name + '_' + model['category'] + f"_h={model['h']:.2f}_kappa={model['kappa']}_L={model['L']}_W={model['W']}"
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

    # 2D PEC exception: currently useless
    if L == 0 and W == 0:
        return kwant.wraparound.wraparound(sys).finalized()
    return sys.finalized()
def HXY(sys):
    H = sys.hamiltonian_submatrix(sparse=False)
    X, Y = np.zeros(H.shape), np.zeros(H.shape)
    for i, site in enumerate(sys.sites):
        X[i, i] = site.pos[0]
        Y[i, i] = site.pos[1]
    return H, X, Y, np.shape(H)[0]
def spectral_localizer(sys, x, y, e):
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
        H, X, Y, dim = HXY(sys)

    L = kron(sz, H-e*np.identity(dim)) + kappa * (kron(sx, X-x*np.identity(dim)) + kron(sy, Y-y*np.identity(dim)))
    return L, dim
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

def spectral_localizer1(sys, x, y, e):
    kappa = model['kappa']

    H = sys.hamiltonian_submatrix()
    dim = np.shape(H)[0]
    X, Y = np.zeros(H.shape), np.zeros(H.shape)
    for i, site in enumerate(sys.sites):
        X[i, i] = site.pos[0]
        Y[i, i] = site.pos[1]

    L = kron(sz, Y-y*np.identity(dim)) + kappa * (kron(sx, X-x*np.identity(dim)) + kron(sy, H-e*np.identity(dim)))
    return L, dim
  
def eigenvalues_change(sys, e):
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
        if para['x_min'] == para['x_max']:
            L, dim = spectral_localizer(sys, y = y, x = coord_fix, e = e)
        elif para['y_min'] == para['y_max']:
            L, dim = spectral_localizer(sys, y = coord_fix, x = y, e = e)
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
    plt.figtext(0.5, 0.92, xlabel+f', e={e}', ha="center", va="top", fontsize=10, color="blue")
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
                      m = 0.0, t1 = 1.0, h = 0.8,
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
                             y_min = -y, y_max = 0, num_cc=200,
                             x_min = 0, x_max = 0,
                             num_eigvals = 20,))
        elif model['name'] == DEFECT: # x_fix = 0
            # y_min = -const*cc, y_max = const*cc
            para.update(dict(func = func,
                             y_min = 0, y_max = 0, num_cc=200,
                             x_min = -x, x_max = 0,
                             num_eigvals = 20,))
    else:
        print('change_para error')
        system.exit()

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

#current_Jr(DEFECT, SINGLE, max_E=0.175)

def draw_H_psi():
    change_model(DEFECT, SINGLE)
    for i in range(10):
        for h in [0.0]:
            model['h'] = h
            sys = model_builder()
            H = sys.hamiltonian_submatrix(sparse=False)
            evals, evecs = eigh(H)
            def custom_sort(evals, evecs):
                indices = np.argsort(np.abs(evals))
                sorted_evals = evals[indices]
                sorted_eigens = evecs[:, indices]
                return sorted_evals, sorted_eigens
            sorted_evals, sorted_evecs = custom_sort(evals, evecs)
            
            data = np.abs(sorted_evecs[:, i*2])**2
            vmax = np.max(data)
            fig, ax = plt.subplots()
            kwant.plotter.map(sys, data, vmax=vmax, ax=ax)
            kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
            label = pick_label(model['name'])
            plt.title(f"h={h} |e|={abs(sorted_evals[i*2])} order={i}")
            plt.savefig(f"/content/LDOS_{label}_{i}.png")
            plt.show()

def draw_L_psi(sys, x=0, y=0, e=0):
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
    L, dim = spectral_localizer(sys, x=x, y=y, e=e)
    np.random.seed(0)
    eval, evec = eigsh(L, k=1, which='SM', return_eigenvectors=True)#, tol=1e-10)
    psi1, psi2 = evec[:dim, 0], evec[dim:, 0]
    psi11, psi12, psi21, psi22 = subvec(evec)
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
        plt.ylim(-3.5, -1)
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

def perturbation_kappa(sys, x, y, epsilon):
    H, X, Y, dim = HXY(sys)
    model['kappa'] = 0
    L0, _ = spectral_localizer(sys, x, y, 0)
    evals, evecs = eigh(H)
    H_evals, H_evecs = custom_sort(evals, evecs)
    # 取最小的 +a(v 0) +a(0 w) -a(w 0) -a(0 v)
    if H_evals[0] > 0:
        v = H_evecs[:, 0]
        w = H_evecs[:, 1]
    else:
        v = H_evecs[:, 1]
        w = H_evecs[:, 0]
    # psi1 = np.concatenate((v, np.zeros(dim)))
    # psi2 = np.concatenate((np.zeros(dim), w))
    psi1 = np.concatenate((v, w))/sqrt(2)
    psi2 = np.concatenate((v, -w))/sqrt(2)
    # print(np.conj(psi1).T@psi2)

    model['kappa'] = epsilon
    L, _ = spectral_localizer(sys, x, y, 0)
    V = L-L0

    V_deg = np.array([
    [np.conj(psi1).T @ V @ psi1, np.conj(psi1).T @ V @ psi2],
    [np.conj(psi2).T @ V @ psi1, np.conj(psi2).T @ V @ psi2]])

    V_evals, V_evecs = eigh(V_deg)

    print("first-order localizer corrections: ", V_evals)

    print("New eigenstates:")
    for i, eigenvector in enumerate(V_evecs.T):
        print(f"Eigenstate {i+1}: {eigenvector}")


current_Jr(DEFECT, SINGLE)

# change_model(HALDANE, NONE)
# change_para(CHANGE)
# model['h'] = 1.0
# sys = model_builder()
# # for kappa in [5, 3, 2, 1, 0.5, 0.1]:
# #     model['kappa'] = kappa
# #     eigenvalues_change(sys, 0)
# model['kappa'] = 2.0
# # eigenvalues_change(sys, 0)
# perturbation(sys, x=0.0, y=-2.87108, epsilon_x=0, epsilon_y=0.1)
# draw_L_psi(sys, x=0.0, y=-2.87108)

# change_model(DEFECT, SINGLE)
# change_para(GAP)
# model['h'] = 0.8
# sys = model_builder()
# for kappa in [5, 3, 2, 1, 0.5, 0.1, 0.05, 0.01]:
#     model['kappa'] = kappa
#     localizer_gap(sys, 0)
# # model['kappa'] = 1.0
# # perturbation(sys, x=0.0, y=0.0, epsilon_x=0.1, epsilon_y=0)
# # draw_L_psi(sys, x=-0.0, y=0.0)

# h=1 kappa=2 x=0 y=-2.8711; h=1.1953 kappa=1 x=0 y=0