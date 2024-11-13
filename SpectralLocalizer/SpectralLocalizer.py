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
from math import sqrt, pi, sin, cos, dist
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

SSH, HALDANE, HALDANETRI, DEFECT, PYBINDING = 'SSH', 'haldane', 'haldane and triangular', 'defect graphene', 'pybinding'
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
    if model['name'] == SSH:
        return ssh()
    # return haldane_triangular_pybinding()

def ssh():
    lat = kwant.lattice.chain(norbs=1)
    sys = kwant.Builder()
    L, t1, t2 = model['L'], model['t1'], model['t2']
    for i in range(2*L):
        sys[lat(i)] = 0

    for i in range(2*L-1):
        if i % 2 == 0: # A->B 弱
            sys[lat(i), lat(i + 1)] = -t1
        else: # B->A 强
            sys[lat(i), lat(i + 1)] = -t2
    
    #kwant.plot(sys)
    return sys.finalized()

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
    
    def find_distribution(sys):
        H = sys.hamiltonian_submatrix(sparse=False)
        evals, _ = eigh(H)
        np.set_printoptions(suppress=True, precision=7) 
        print(evals[(evals >= -0.002) & (evals <= 0)])
        plt.hist(evals, bins=20, range=(-0.2, 0.2), color='blue', alpha=0.7)
        tick_marks = np.linspace(-0.2, 0.2, 21) 
        plt.xticks(tick_marks, rotation=45, fontsize=7) 
        plt.title(f"Energy Frequency Distribution: h={model['h']} L={model['L']}")
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.ylim(0, 10)
        plt.grid(True)
        plt.show()
        plt.close()
    
    def draw_distribution():
        h_num, energy_num = 1200, 15
        h_list = np.linspace(0.3, 1.5, h_num)
        energy_list = np.zeros((energy_num, h_num))
        for i, h in enumerate(h_list):
            model['h'] = h
            sys = model_builder()
            H = sys.hamiltonian_submatrix(sparse=False)
            #evals = eigsh(H, k=energy_num*3, sigma=0, return_eigenvectors=False, tol=1e-5)
            evals, _ = eigh(H)
            negative_evals = evals < 0
            filtered_evals = evals[negative_evals]
            final_evals = sorted(filtered_evals, key=abs)
            energy_list[:, i] = final_evals[0:energy_num]
        energy_list = -energy_list
        cutoff = 0.1
        plt.figure()
        plt.xlabel('h')
        plt.ylabel('energy')
        plt.ylim(0, cutoff)
        plt.title(f"L={model['L']}")
        lines = []
        legends = []
        for n in range(energy_num):
            if any(energy_list[n] < cutoff):
                plt.scatter(h_list, energy_list[n], label=f'{n}', s=1)
        plt.legend()
        plt.show()
    
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
            start, stop = args[0], args[1]
            way = f"cutoff_start={start:.2f}_stop={stop:.2f}"
            for i, e in enumerate(evals):
                if -e > start and -e <= stop:
                    if sum_current is None:
                        sum_current = current[i]
                    else:
                        sum_current += current[i]
            
        elif tag == GAUSSIAN:
            start, stop, min, max, sigma = args[0], args[1], args[2], args[3], args[4]
            way = f"gaussian_start={start:.2f}_stop={stop:.2f}_sigma={sigma:.2f}"
            gaussian_values = np.exp(-(evals - stop)**2 / (2 * sigma**2)) + np.exp(-(evals - start)**2 / (2 * sigma**2))
            gaussian_values /= np.max(gaussian_values)
            gaussian_values[(evals >= start) & (evals <= stop)] = 1
            if min is not None:
                way += f"_min={min:.2f}"
                gaussian_values[evals <= min] = 0
            if max is not None:
                way += f"_max={max:.2f}"
                gaussian_values[evals >= max] = 0

            for i, e in enumerate(gaussian_values):
                if e > 0:
                    if sum_current is None:
                        sum_current = e * current[i]
                    else:
                        sum_current += e * current[i]

        elif tag == STATE:
            start, stop = args[0], args[1]
            cutoff = args[2]
            negative_evals = evals < 0
            filtered_evals = evals[negative_evals]
            filtered_current = current[negative_evals, :]
            way = f"state_start={start}_stop={stop}_cutoff={cutoff}"
            sorted_evals, sorted_current = custom_sort(filtered_evals, filtered_current, True)
            if cutoff is not None:
                for i, e in enumerate(sorted_evals):
                    if abs(e) > cutoff:
                        sorted_current[i, :] = 0 
            sum_current = sorted_current[start]
            for i in range(start+1, stop):
                sum_current += sorted_current[i]
        return way, sum_current
    
    def draw_current():
        h_list = [0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7]
        bounds = [(-2.5, 2.5), (-2.5, 2.5)]
        for h in h_list:
            model['h'] = h
            sys = model_builder()
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            plt.xlim(bounds[0][0], bounds[0][1])
            plt.ylim(bounds[1][0], bounds[1][1])
            #signature_change3(sys, e=0, bounds=bounds)
            evals, current = pure_current_info(sys)
            way, sum_current = current_filter(evals, current, GAUSSIAN, 0, 0, None, 0, 1/13)
            index = -1
            for head, tail in sys.graph:
                index += 1
                if abs(sum_current[index]) < 0.00001:
                    continue
                start_point, end_point = sys.sites[head].pos, sys.sites[tail].pos
                #x_start, y_start, x_end, y_end = round(p1[0], 3), round(p1[1], 3), round(p2[0], 3), round(p2[1], 3)
                if np.linalg.norm(start_point) > 4.5:
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
                arrow_length = normalized * weight * 40
                if np.linalg.norm(arrow_length) < 0.16:
                    continue

                if np.linalg.norm(start_point) > 0.8:
                    mutation_scale = 10
                    color = 'red'
                    alpha = 1
                else:
                    mutation_scale = 8
                    color = 'blue'
                    alpha = 0.2
                arrow = patches.FancyArrowPatch(start_point, start_point+arrow_length,
                                                arrowstyle='-|>', connectionstyle='arc3,rad=0.0', 
                                                mutation_scale=mutation_scale, color=color, alpha=alpha)
                ax.add_patch(arrow)
                # if 0<=mid_x<=3 and 0<=mid_y<=3:
                #     ax.text(mid_x, mid_y, f'{weight:.4f}'.lstrip('0').replace('-0.', '-.'), color='red', fontsize=8, ha='center', va='center')
            kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.6, 0.7, 1.0, 0.3))
            # 隐藏x轴和y轴的刻度标签
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # 隐藏x轴和y轴的刻度
            ax.set_xticks([])
            ax.set_yticks([])
            plt.title(f"h={h}")
            plt.tight_layout()
            plt.savefig(f"\content\h{h:.2f}.png", bbox_inches='tight')
            plt.show()
            plt.close()
    
    def magnitude_info(sum_current, buckets=np.array([])):
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
        sum_buckets = np.zeros(len(buckets) + 1)
        for key, value in sort_distance.items():
            tmp = np.sum(temp[value])
            sum_direct += tmp
            sum_angular += tmp*key
            index = np.digitize(key, buckets)
            sum_buckets[index] += tmp
            magnitude[value] = tmp/len(temp[value])
            
        return magnitude, sum_direct, sum_angular, sum_buckets
    
    change_model(name, category)
    model['L'] = model['W'] = 25
    def pos():
        temp_sys = model_builder()
        distance, pos_info = edge_info(temp_sys)
        sort_r = sorted(distance.keys())
        sort_distance = {key: index for index, key in enumerate(sort_r)}
        return pos_info, sort_r, sort_distance
    pos_info, sort_r, sort_distance = pos()
    #r是实际距离 sort_r里是从小到大的r sort_distance里是从小到大的r和对应的编号
    # for L in [9, 13, 17, 21, 25]:
    #     model['L'] = model['W'] = L
    #     sys = model_builder()
    #     find_distribution(sys)
        # draw_distribution()
    #draw_current()
    
    def draw_h_fixed(whichsum):
        h_list = [0.7, 1.3]
        for h in h_list:
            model['h'] = h
            sys = model_builder()
            label = pick_label(model['name'])
            evals, current = pure_current_info(sys)
            way, sum_current = current_filter(evals, current, GAUSSIAN, 0, 0, None, 0, 1/13)
            magnitude, _, _, _ = magnitude_info(sum_current)
            if whichsum == 'J(r)':
                multiply_item=magnitude
            elif whichsum == 'J(r)_r':    
                multiply_item = magnitude * sort_r
            
            plt.figure()
            plt.scatter(sort_r, multiply_item, s=1)
            plt.title(f'h={h}')
            plt.plot(sort_r, multiply_item, marker='o')
            plt.axhline(0, color='grey', linewidth=1)
            #plt.ylim(-0.18, 0.1)
            plt.xlabel('r')
            plt.ylabel('Average of J(r)*r') 
            #plt.figtext(0.5, 0.95, f"h={h:.2f} {way}", ha="center", va="top", fontsize=10, color="blue")
            plt.savefig(f'/content/current_J(r)_r_{h}.png')
            plt.show()
            plt.close()
    #draw_h_fixed('J(r)')
    #draw_h_fixed('J(r)_r')

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
        h_num = 180
        #h_list = np.concatenate((np.linspace(0.3, 1.8, h_num), np.linspace(1.8, 9.8, 16)))
        h_list = np.linspace(0.3, 1.8, h_num)
        
        def diff_size():
            nonlocal pos_info, sort_r, sort_distance
            plt.figure()
            for L in [9, 13, 17, 21, 25]:
                model['L'] = L
                model['W'] = L
                pos_info, sort_r, sort_distance = pos()
                sum_list = []
                for i, h in enumerate(h_list):
                    model['h'] = h
                    sys = model_builder()
                    evals, current = pure_current_info(sys)
                    if tag == GAUSSIAN:
                        way, sum_current = current_filter(evals, current, tag, 0, 0, None, 0, 1/L)
                    if whichsum == 'J(r)':
                        _, sum, _, _ = magnitude_info(sum_current)
                    elif whichsum == 'J(r)_r':    
                        _, _, sum, _ = magnitude_info(sum_current)
                    # if sum > 0 and sum_list[i-1] < 0:
                    #     zero_points = [h_list[i-1], h]
                    sum_list.append(sum)
                plt.plot(h_list, sum_list, label=f'L={L}')
            plt.axhline(0, color='grey', linewidth=1)
            plt.xlabel('h')
            plt.legend()
            plt.ylabel('J')
            # plt.figtext(0.5, 0.92, f"cross:[{zero_points[0]:.3f}, {zero_points[1]:.3f}]", ha="center", va="top", fontsize=10, color="blue")
            # plt.figtext(0.5, 0.97, way, ha="center", va="top", fontsize=10, color="blue")
            # plt.savefig(f'/content/sum_{whichsum}_{way}_{label}.png')
            plt.show()
            plt.close()
        
        def diff_nonsize():
            #buckets = np.array([0.6, 1.6, 3, 5])
            #buckets_num = len(buckets)+1
            if tag == STATE:
                line_num = 10
                cutoff = 0.1
            elif tag == GAUSSIAN:
                line_num = 1

            # STORE DATA
            # if model['L'] == 25:
            #     pack_storage = 300
            #     tmp_storage = np.zeros((pack_storage, line_num, 3888))
            # READ DATA
            if model['L'] == 25:
                file_content = np.zeros((h_num, line_num, 3888))
                pack_storage = 300
                file_path = '/content/drive/My Drive/Colab Notebooks/'
                for n in range(5):
                    data = np.load(file_path+f'data{n}.npz')
                    file_content[n*pack_storage:(n+1)*pack_storage] = data['data']
                    data.close()

            sum_list = np.zeros((line_num, h_num))#, buckets_num))
            for i, h in enumerate(h_list):
                #if i >= 3*pack_storage and i < 4*pack_storage:

                # model['h'] = h
                # sys = model_builder()
                # evals, current = pure_current_info(sys)
                for n in range(line_num):
                #     if tag == STATE:
                #         way, sum_current = current_filter(evals, current, tag, n, n+1, cutoff)
                #     elif tag == GAUSSIAN:
                #         way, sum_current = current_filter(evals, current, tag, 0, 0, None, 0.2, 0.05)
                    sum_current = file_content[i, n]
            #         tmp_storage[i%pack_storage, n, :] = sum_current
            # file_path = '/content/drive/My Drive/Colab Notebooks/'
            # np.savez(file_path+'data3.npz', data=tmp_storage)
                    if whichsum == 'J(r)':
                        _, sum, _, sum_buckets = magnitude_info(sum_current)#, buckets)
                    elif whichsum == 'J(r)_r':    
                        _, _, sum, sum_buckets = magnitude_info(sum_current)#, buckets)
                    #sum_list[n, i, :] = sum_buckets
                    sum_list[n, i] = sum
            def single_sum():
                plt.figure()
                for n in range(line_num):
                    sum_masked = np.ma.masked_where(sum_list[n] == 0, sum_list[n])
                    if np.any(sum_masked.mask == False):
                        plt.scatter(h_list, sum_masked, label=f"{n}", s=1)
                plt.axhline(0, color='grey', linewidth=1)
                #plt.ylim(-0.05, 0.1)
                plt.xlabel('h')
                plt.legend()
                plt.ylabel('sum of '+whichsum)
                # plt.figtext(0.5, 0.92, f"cross:[{zero_points[0]:.3f}, {zero_points[1]:.3f}]", ha="center", va="top", fontsize=10, color="blue")
                #plt.figtext(0.5, 0.97, way, ha="center", va="top", fontsize=10, color="blue")
                # plt.savefig(f'/content/sum_{whichsum}_{way}_{label}.png')
                plt.title(f"L={model['L']}")
                plt.show()
                plt.close()
            def multiple_sum_r():
                for r in range(buckets_num):
                    left, right = 'origin', 'outside'
                    if r != 0:
                        left = buckets[r-1]
                    if r != len(buckets):
                        right = buckets[r]
                    plt.figure()
                    for n in range(line_num):
                        plt.scatter(h_list, sum_list[n, :, r], label=f"{n}", s=2)
                    plt.axhline(0, color='grey', linewidth=1)
                    plt.ylim(-0.5, 0.65)
                    plt.xlabel('h')
                    plt.legend()
                    plt.ylabel('sum of '+whichsum)
                    # plt.figtext(0.5, 0.92, f"cross:[{zero_points[0]:.3f}, {zero_points[1]:.3f}]", ha="center", va="top", fontsize=10, color="blue")
                    plt.figtext(0.5, 0.97, f"[{left}, {right})", ha="center", va="top", fontsize=10, color="blue")
                    # plt.savefig(f'/content/sum_{whichsum}_{way}_{label}.png')
                    plt.show()
                    plt.close()
            single_sum()
            #multiple_sum_r()

        diff_size()
        #diff_nonsize()

    #draw_sum('J(r)', STATE)
    #draw_sum('J(r)', GAUSSIAN)
    
    def plot_hc():
        label = ['J', 'J\'']
        nonlocal pos_info, sort_r, sort_distance
        plt.figure()
        L_list = [9, 13, 17, 21, 25]
        sum_list = np.zeros((2, 5))
        for l, L in enumerate(L_list):
            model['L'] = L
            model['W'] = L
            pos_info, sort_r, sort_distance = pos()
            for line in range(2):
                def brentq_search(h):
                    model['h'] = h
                    sys = model_builder()
                    evals, current = pure_current_info(sys)
                    way, sum_current = current_filter(evals, current, GAUSSIAN, 0, 0, None, 0, 1/L)
                    if line == 0:
                        _, sum, _, _ = magnitude_info(sum_current)
                    else:
                        _, _, sum, _ = magnitude_info(sum_current)
                    return sum
                zero = brentq(brentq_search, 0.3, 1.8, xtol=1e-4)
                sum_list[line, l] = zero
        for line in range(2):
            plt.plot(L_list, sum_list[line, :], label=label[line])
        plt.legend()
        plt.xticks(L_list)
        plt.xlabel('L')
        plt.ylabel('hc')
        plt.ylim(0, 2)
        plt.show()
        print(sum_list)
    #plot_hc()

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
                _, sum, _, _ = magnitude_info(sum_current)
            elif whichsum == 'J(r)_r':    
                _, _, sum, _ = magnitude_info(sum_current)
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

import h5py

class DataStorage:
    def __init__(self, position_file='position_data.h5', energy_file='energy_data.h5', current_file='current_data.h5',
                 num_edges=2, num_h=150, num_energies=2):
        self.position_file = position_file
        self.energy_file = energy_file
        self.current_file = current_file
        self.num_edges = num_edges
        self.num_h = num_h
        self.num_energies = num_energies
    
    def write_positions(self, positions):
        with h5py.File(self.position_file, 'a') as position_file: 
            if 'positions' not in position_file:
                position_file.create_dataset('positions', data=positions)
            else:
                position_file['positions'][...] = positions
    
    def read_positions(self):
        with h5py.File(self.position_file, 'r') as position_file:
            positions_data = position_file['positions'][:]
            self.num_edges = positions_data.shape[0]
        return positions_data
    
    def write_energies(self, energies, h_index):
        with h5py.File(self.energy_file, 'a') as energy_file:
            if 'energies' not in energy_file:
                energy_data = energy_file.create_dataset('energies', (self.num_h, self.num_energies), dtype='float64')
            else:
                energy_data = energy_file['energies']
            energy_data[h_index] = energies
    
    def read_energies(self, h_index):
        with h5py.File(self.energy_file, 'r') as energy_file:
            energies_data = energy_file['energies'][h_index]
            self.num_energies = energies_data.shape[0]
        return energies_data

    def write_currents(self, currents, h_index):
        with h5py.File(self.current_file, 'a') as current_file:
            if 'currents' not in current_file:
                current_data = current_file.create_dataset('currents', (self.num_edges, self.num_h, self.num_energies), dtype='float64')
            else:
                current_data = current_file['currents']
            current_data[:, h_index, :] = currents
    
    def read_currents(self, h_index):
        with h5py.File(self.current_file, 'r') as current_file:
            current_data = current_file['currents'][:, h_index, :]
        return current_data

def write_data():
    change_model(DEFECT, SINGLE)
    model['L'] = model['W'] = 13
    sys = model_builder()
    H = sys.hamiltonian_submatrix(sparse=False)
    num_h = 150
    num_energies = np.shape(H)[0]
    
    #存位置
    positions_list = []
    for head, tail in sys.graph:
        positions_list.append([sys.sites[head].pos, sys.sites[tail].pos])
    positions = np.array(positions_list)
    num_edges = positions.shape[0]
    storage = DataStorage(num_edges=num_edges, num_h=num_h, num_energies=num_energies)
    storage.write_positions(positions)
    
    h_list = np.linspace(0.3, 1.8, num_h)
    for i, h in enumerate(h_list):
        model['h'] = h
        sys = model_builder()
        
        #存能量
        H = sys.hamiltonian_submatrix(sparse=False)
        evals, evecs = eigh(H)
        sorted_indices = np.argsort(np.abs(evals))
        sorted_evals = evals[sorted_indices]
        sorted_evecs = evecs[:, sorted_indices]
        storage.write_energies(sorted_evals, h_index=i)
        
        #存电流
        J = kwant.operator.Current(sys)
        currents = np.array([J(sorted_evecs[:, j]) for j in range(num_energies)]).T
        storage.write_currents(currents, h_index=i)
        
    # #test
    # print(storage.read_currents(0))
    # print(storage.read_currents(1))

def read_data():
    def get_interactions(positions, k=0, threshold=0):
        if k == 0:
            line_vector = np.array([0, 1])
        else:    
            line_vector = np.array([-1, 1/k])
        interact = np.zeros(storage.num_edges)
        signs = np.zeros(storage.num_edges)
        for i, edge in enumerate(positions):
            (x1, y1), (x2, y2) = edge
            if np.sqrt(((x1+x2)/2)**2 + ((y1+y2)/2)**2) < threshold: #距离小的不要
                continue
            if (x1<0 and y1<0) or (x2<0 and y2<0): #不在第一象限的不要（一条边的例外）
                continue
            head_in_first_quadrant = (x1 >= 0) and (y1 >= 0)
            tail_in_first_quadrant = (x2 >= 0) and (y2 >= 0)
            if not (head_in_first_quadrant or tail_in_first_quadrant): #不在第一象限不要
                continue
            y1_line = k * x1
            y2_line = k * x2
            if (y1 > y1_line and y2 < y2_line) or (y1 < y1_line and y2 > y2_line): #有交点
                #print(x1,y1,x2,y2)
                edge_vector = np.array([x2 - x1, y2 - y1])/np.linalg.norm([x2 - x1, y2 - y1])
                interact[i] = np.dot(edge_vector, line_vector) / np.linalg.norm(line_vector)
                signs[i] = np.sign(interact[i])
        return interact, signs
    
    def get_sumcurrents():
        sum_currents = np.zeros((storage.num_h, storage.num_edges))
        for h_i in range(storage.num_h):
            energies = storage.read_energies(h_i)
            gaussian_values = np.exp(-energies**2 / (2 * sigma**2))
            gaussian_values /= np.max(gaussian_values)
            gaussian_values[energies>=0] = 0
            currents = storage.read_currents(h_i)
            sum_currents[h_i] = np.array([np.dot(currents[j], gaussian_values) for j in range(storage.num_edges)])
        return sum_currents
        
    sigma = 1/13
    storage = DataStorage()
    storage.num_h = 150
    positions = storage.read_positions()
    h_list = np.linspace(0.3, 1.8, storage.num_h)
    sum_currents = get_sumcurrents()
    
    plt.figure()
    plt.axhline(0, color='gray', linestyle='-', linewidth=1)
    for l in [0,1,2,3,4,5,6]:
        for r in [0.6]:
            interactions, signs = get_interactions(positions, k=l, threshold=r)
            flow_list = np.dot(sum_currents, interactions)
            i = np.where(flow_list < 0)[0][-1]
            plt.plot(h_list, flow_list, label=f'interact k={l:.3f} r={r:.3f} zero={h_list[i]:.3f}')
            # flow_list = np.dot(sum_currents, signs)
            # i = np.where(flow_list < 0)[0][-1]
            # plt.plot(h_list, flow_list, label=f'signs k={l:.3f} r={r:.3f} zero={(h_list[i]+h_list[i+1])/2:.3f}')
            # #test
            # nonzero_indices = np.nonzero(signs)[0]
            # for i in nonzero_indices:
            #     print(l, f'edge={positions[i]}')
            #     print(f'signs={signs[i]}')
            #     print(f'h=0.3 sum_currents={sum_currents[0, i]}')
            #     print('\n')
        
    plt.legend()
    plt.show()

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

def pick_label(name, ifh = True, ifkappa = True):
    if name == HALDANETRI:
        return name + '_' + model['category'] + '_' + f"t3={model['t3']:.2f}_h={model['h']:.2f}_t2={model['t2']:.2f}_L={model['L']}_W={model['W']}"
    if name == DEFECT or name == HALDANE:
        if ifh == False:
            return name + '_' + model['category'] + f"_kappa={model['kappa']}_L={model['L']}_W={model['W']}"    
        if ifkappa == False:
            return name + '_' + model['category'] + f"_h={model['h']:.2f}_L={model['L']}_W={model['W']}"    
        return name + '_' + model['category'] + f"_h={model['h']:.2f}_kappa={model['kappa']}_L={model['L']}_W={model['W']}"
    if name == SSH:
        return name+f"_t1={model['t1']}_t2={model['t2']}_L={model['L']}_kappa={model['kappa']}"
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
def spectral_localizer_1d(sys, x, e, *args):
    kappa = model['kappa']
    H = sys.hamiltonian_submatrix(sparse=False)
    dim = np.shape(H)[0]
    X = np.zeros(H.shape)
    for i, site in enumerate(sys.sites):
        X[i, i] = site.pos[0]
    
    if args:
        P = X-x*np.identity(dim)
        P_diag = np.diag(P)
        if args[0] == 'abs':
            P_diag_edit = np.sign(P_diag) * (np.abs(P_diag) ** args[1])
            P_edit = np.diag(P_diag_edit)
        else:
            print('only abs accepted')
        L = kappa*kron(sx, P_edit)+kron(sy, H-e*np.identity(dim))
    else:
        L = kappa*kron(sx, X-x*np.identity(dim))+kron(sy, H-e*np.identity(dim))
    return L, dim
def spectral_localizer_2d(sys, x, y, e, *args):
    kappa = model['kappa']
    def pybinding_case():
        H = sys.hamiltonian.toarray(model)
        dim = np.shape(H)[0]
        X, Y = np.zeros(H.shape), np.zeros(H.shape)
        for i, site in enumerate(sys.system.positions.x):
            X[i, i] = site
        for i, site in enumerate(sys.system.positions.y):
            Y[i, i] = site
    H, X, Y, dim = HXY(sys)

    if args:
        P = X-x*np.identity(dim) - 1j * (Y-y*np.identity(dim))
        P_diag = np.diag(P)
        if args[0] == 'abs':
            P_diag_edit = [(z / np.abs(z) * (np.abs(z) ** args[1])) if np.abs(z) != 0 else 0 for z in P_diag]
            P_edit = np.diag(P_diag_edit)
        else:
            print('only abs accepted')
        L = np.block([
            [H-e*np.identity(dim), kappa*P_edit],
            [kappa*np.conj(P_edit), -H+e*np.identity(dim)]
        ])
    else:
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
        if model['name'] == SSH:
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
                          t1=0.5, t2=1,
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

#write_data()
read_data()   

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