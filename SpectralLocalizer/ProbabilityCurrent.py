from __future__ import division
import kwant
import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from numpy.linalg import eigh
from scipy.linalg import kron
# import pylab as py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys as system
import cmath
from math import sqrt, pi, sin, cos, dist
import warnings
from matplotlib.lines import Line2D

import KwantModel as km

CONTINUEPOINT = 0.1

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
def current_Jr(name, category, ax=None, h=0):
    CUTOFF, GAUSSIAN, STATE = 'max_E', 'gaussian', 'state'

    def edge_info(sys):
        distance = {}
        pos_info = [] # [[dot1, r1], [dot2, r2]]
        r_index = 0
        for tail, head in sys.graph:
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
    
    def draw_current(ax, h):
        bounds = [(-2.6, 2.6), (-2.5, 2.5)]
        km.model['h'] = h
        sys = km.model_builder()
        ax.set_aspect('equal')
        plt.xlim(bounds[0][0], bounds[0][1])
        plt.ylim(bounds[1][0], bounds[1][1])
        kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.5, 0.5, 0.5, 0.3),hop_lw=0.07)
        evals, current = pure_current_info(sys)
        way, sum_current = current_filter(evals, current, GAUSSIAN, 0, 0, None, 0, 2/km.model['L'])
        index = -1
        for tail, head in sys.graph:
            index += 1
            if abs(sum_current[index]) < 0.0001:
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
            arrow_length = normalized * weight * 55
            if np.linalg.norm(arrow_length) < 0.2:
                continue

            if np.linalg.norm(start_point) > 0.8:
                mutation_scale = 5
                color = 'red'
                alpha = 1
            else:
                mutation_scale = 5
                color = 'black'
                alpha = 0.2
            arrow = patches.FancyArrowPatch(start_point, start_point+arrow_length,
                                            arrowstyle='-|>', connectionstyle='arc3,rad=0.0', 
                                            mutation_scale=mutation_scale, color=color, alpha=alpha, linewidth=0.8)
            ax.add_patch(arrow)
            # if 0<=mid_x<=3 and 0<=mid_y<=3:
            #     ax.text(mid_x, mid_y, f'{weight:.4f}'.lstrip('0').replace('-0.', '-.'), color='red', fontsize=8, ha='center', va='center')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.975, 0.82, rf"$h={h}$",transform=ax.transAxes,ha='right', va='bottom',fontsize=10)
    
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
    
    km.change_model(name, category)
    km.model['L'] = km.model['W'] = 25
    def pos():
        temp_sys = km.model_builder()
        distance, pos_info = edge_info(temp_sys)
        sort_r = sorted(distance.keys())
        sort_distance = {key: index for index, key in enumerate(sort_r)}
        return pos_info, sort_r, sort_distance
    pos_info, sort_r, sort_distance = pos()
    # for L in [9, 13, 17, 21, 25]:
    #     model['L'] = model['W'] = L
    #     sys = model_builder()
    #     find_distribution(sys)
        # draw_distribution()
    draw_current(ax, h)
    
    def draw_h_fixed(whichsum):
        h_list = [0.7, 1.3]
        colors = ['blueviolet', 'darkorange']
        plt.figure()
        plt.axhline(0, color='grey', linewidth=1, linestyle='--',alpha=0.5)
        for i, h in enumerate(h_list):
            km.model['h'] = h
            sys = km.model_builder()
            evals, current = pure_current_info(sys)
            way, sum_current = current_filter(evals, current, GAUSSIAN, 0, 0, None, 0, 2/km.model['L'])
            magnitude, _, _, _ = magnitude_info(sum_current)
            if whichsum == 'J(r)':
                multiply_item=magnitude
            elif whichsum == 'J(r)_r':
                multiply_item = magnitude * sort_r
            plt.plot(sort_r, multiply_item, marker='o', linestyle='-.', markersize=3, label=rf"$h={h}$", color=colors[i], alpha=0.8)
        plt.xlim(0, 6)
        plt.xlabel(r'$r$')
        plt.ylabel(r'$\mathcal{J}(r)$') 
        plt.legend(loc='lower right',frameon=False)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both')
        ax.minorticks_on()
        plt.ylim(-0.02, 0.03)
        dot_positions = [
            (1.323,r'$d$'),
            (0.866,r'$c$'),
            (1.5,r'$e$'),
            (0.5,r'$b$'),
            (0.289, r'$a$')]
        for (x, text) in dot_positions:
            ax.plot(x, -0.02, 'o', markerfacecolor='black', markeredgecolor='black', markersize=3,clip_on=False)
            ax.text(x, -0.0185, text, fontsize=11, ha='center', va='center',clip_on=False)
        
        ax_inset = inset_axes(ax, width="50%", height="50%",loc='upper right')
        image = mpimg.imread("/Users/ruiqi/Documents/tmp/currents/r.png")
        ax_inset.imshow(image)
        ax_inset.axis("off")
        plt.savefig(f"/Users/ruiqi/Documents/tmp/currents/fig4a.png",dpi=300, bbox_inches='tight')
        #plt.show()
        
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
    def __init__(self, file_path, position_file='position_data.h5', energy_file='energy_data.h5', current_file='current_data.h5',
                 num_edges=2, num_h=150, num_energies=2):
        self.file_path = file_path
        self.position_file = self.file_path+position_file
        self.energy_file = self.file_path+energy_file
        self.current_file = self.file_path+current_file
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

def test_triangle():
    lat = kwant.lattice.general([(1, 0), (0.5, np.sqrt(3)/2)], name='tri', norbs=1)
    sys = kwant.Builder()
    A = (0, 0) 
    B = (1, 0) 
    C = (0, 1)
    sys[lat(*A)] = 0
    sys[lat(*B)] = 0
    sys[lat(*C)] = 0
    sys[lat(*A), lat(*B)] = 1j
    sys[lat(*B), lat(*C)] = 1j
    sys[lat(*C), lat(*A)] = 1j
    
    for i in sys.hopping_value_pairs():
        print(i)
    print('...')
    for hopping in sys.hoppings():
        print(hopping)
    sys = sys.finalized()
    
    H = sys.hamiltonian_submatrix(sparse=False)
    print(H)
    for i, site in enumerate(sys.sites):
        print(i, site.pos)
    evals, evecs = eigh(H)
    print(evals, evecs)
    
    for tail, head in sys.graph:
        print(sys.sites[tail].pos, sys.sites[head].pos)
    print(sys.graph.head(5), 'head')
    print(sys.graph.all_edge_ids(0, 2), 'all edge ids')
    J = kwant.operator.Current(sys)
    psi = evecs[:, 0]
    print('\n', psi)
    currents = J(psi)
    print(currents)
    #kwant.plotter.current(sys, currents)
    
    print('\n test')
    theta_a = np.angle(psi[0])
    theta_c = np.angle(psi[1])
    theta_ac = (theta_a - theta_c) % (2 * np.pi)
    print(theta_ac, f'should be {2*pi/3}')
    print('A->C', f"J gives {-currents[1]}", f"theory gives {np.sin(theta_ac+pi/2)}")
    
def test_current_direction():
    km.change_model(DEFECT, SINGLE)
    km.model['m']=0
    km.model['L'] = km.model['W'] = 1
    km.model['h'] = 1
    sys = test_triangle()
    J = kwant.operator.Current(sys)
    H = sys.hamiltonian_submatrix(sparse=False)
    # for site in sys.sites:
    #     print(site.pos)
    evals, evecs = eigh(H)
    evals, evecs = custom_sort(evals, evecs, False)
    for tail, head in sys.graph:
        print(sys.sites[head].pos, sys.sites[tail].pos)
    for i, val in enumerate(evals):
        print(val, ':')
        #print(evecs[:, i])
        print(J(evecs[:, i]))
    
def write_data():
    #change_model(DEFECT, CLUSTER)
    #change_model(HALDANE, NONTRIVIAL)
    km.change_model(km.DEFECT, km.SINGLE)
    km.model['m']=0
    L = km.model['L'] = km.model['W'] = 25
    sys = km.model_builder()
    H = sys.hamiltonian_submatrix(sparse=False)
    num_h = 400
    num_energies = np.shape(H)[0]
    
    #store positions
    positions_list = []
    for head, tail in sys.graph:
        positions_list.append([sys.sites[head].pos, sys.sites[tail].pos])
    positions = np.array(positions_list)
    num_edges = positions.shape[0]
    storage = DataStorage(file_path=f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/', 
                          num_edges=num_edges, num_h=num_h, num_energies=num_energies)
    storage.write_positions(positions)
    
    h_list = np.linspace(0.0, 2.0, num_h)
    for i, h in enumerate(h_list):
        km.model['h'] = h
        sys = km.model_builder()
         
        #store energies
        H = sys.hamiltonian_submatrix(sparse=False)
        evals, evecs = eigh(H)
        sorted_indices = np.argsort(np.abs(evals))
        sorted_evals = evals[sorted_indices]
        sorted_evecs = evecs[:, sorted_indices]
        storage.write_energies(sorted_evals, h_index=i)
        
        #store currents
        J = kwant.operator.Current(sys)
        currents = np.array([J(sorted_evecs[:, j]) for j in range(num_energies)]).T
        storage.write_currents(currents, h_index=i)
        
    # #test
    # print(storage.read_currents(0))
    # print(storage.read_currents(1))

def read_data():
    GAUSSIAN, STATE = "gaussian", "state"
    def get_interactions(storage, positions, k=0, threshold1=None, threshold2=None):
        if k == 0:
            line_vector = np.array([0, 1])
        else:    
            line_vector = np.array([-1, 1/k])
        interact = np.zeros(storage.num_edges)
        signs = np.zeros(storage.num_edges)
        for i, edge in enumerate(positions):
            (x2, y2), (x1, y1) = edge
            if threshold1 is not None:
                if np.sqrt(((x1+x2)/2)**2 + ((y1+y2)/2)**2) < threshold1: #距离小的不要，按中点算
                    continue
            if threshold2 is not None:
                if np.sqrt(((x1+x2)/2)**2 + ((y1+y2)/2)**2) > threshold2: #距离大的不要，按中点算
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
    
    def get_sumcurrents(storage, tag, *args):
        sum_currents = np.zeros((storage.num_h, storage.num_edges))
        if tag == GAUSSIAN:
            sigma = args[0]
            for h_i in range(storage.num_h):
                energies = storage.read_energies(h_i)
                #center = negative_energies[np.argmin(np.abs(negative_energies))]
                gaussian_values = np.exp(-(energies-0)**2 / (2 * sigma**2))
                gaussian_values[energies>=0] = 0
                gaussian_values = gaussian_values*np.sqrt(2/np.pi)/sigma
                
                currents = storage.read_currents(h_i)
                sum_currents[h_i] = np.array([np.dot(currents[j], gaussian_values) for j in range(storage.num_edges)])
        elif tag == STATE:
            state_index = args[0]
            cutoff = args[1]
            for h_i in range(storage.num_h):
                energies = storage.read_energies(h_i)
                negative_indices = np.where(energies < 0)[0]
                negative_energies = energies[negative_indices]
                sorted_indices = np.argsort(np.abs(negative_energies))[:]
                mask = np.zeros_like(energies, dtype=int)
                if state_index < len(sorted_indices):
                    if abs(negative_energies[sorted_indices[state_index]]) <= cutoff:
                        mask[negative_indices[sorted_indices[state_index]]] = 1
                currents = storage.read_currents(h_i)
                sum_currents[h_i] = np.array([np.dot(currents[j], mask) for j in range(storage.num_edges)])
                
        return sum_currents
    
    def hc_plot_exist(ax):
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, 5)][::-1]
        global_hc = np.array([[0.90977444, 0.94486216, 0.9047619,  0.89974937, 0.92982456],
                              [0.93483709, 0.88972431, 0.91478697, 0.89974937, 0.90977444],
                              [0.92982456, 0.89974937, 0.87969925, 0.89473684, 0.9047619 ],
                              [0.90977444, 0.92481203, 0.9047619,  0.89473684, 0.89974937],
                              [0.92982456, 0.89974937, 0.90977444, 0.89974937, 0.89974937],
                              [0.91959799, 0.90954774, 0.88944724, 0.89949749, 0.89949749]])
        L_list = np.array([9, 13, 17, 21, 25, 29], dtype=float)
        a_list = [0.1, 0.5, 1, 2, 4]
        for iL, L in enumerate(L_list):
            for ia, a in enumerate(a_list):
                ax.plot(L_list, global_hc[:, ia], marker='o', linewidth=1, linestyle=':', ms=2, color=colors[ia], alpha=0.6)
        ax.set_xticks(ticks=L_list)
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r'$h_c$')
        ax.set_ylim(0.8, 1)
        ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1])
        ax.grid(alpha=0.5)
        ax.tick_params(direction='in', which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
    def hc_plot(ax):
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, 5)]
        global_hc = np.zeros((6, 5), dtype=float)
        L_list = [9, 13, 17, 21, 25, 29]
        a_list = [0.1, 0.5, 1, 2, 4]
        for iL, L in enumerate(L_list):
            storage = DataStorage(file_path=f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/')
            storage.num_h = 400
            if L == 29:
                storage.num_h = 200
            positions = storage.read_positions()
            h_list = np.linspace(0.0, 2.0, storage.num_h)
            _, signs = get_interactions(storage, positions, k=0, threshold1=None, threshold2=None)
            for ia, a in enumerate(a_list):
                sum_currents = get_sumcurrents(storage, GAUSSIAN, a/L)
                flow_list = np.dot(sum_currents, signs)
                crossing_index = np.where(flow_list > 0)[0][-1]
                if crossing_index == storage.num_h - 1:
                    zero = 0
                else:
                    zero=(h_list[crossing_index]+h_list[crossing_index+1])/2
                global_hc[iL, ia] = zero

        for ia, a in enumerate(a_list):
            ax.plot(L_list, global_hc[:, ia], marker='o', linewidth=1, ms=3.5, color=colors[ia], alpha=0.8)
        ax.set_xticks(ticks=L_list)
        ax.set_xlabel("L", fontsize=8)
        ax.set_ylabel(r'$\text{h}_{\text{c}}$', fontsize=8)
        ax.set_ylim(0.7, 1)
        ax.grid(alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def all_plot():
        L_list = [25]
        a_list = [0.1, 0.5, 1, 2, 4]
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, 5)][::-1]
        for iL, L in enumerate(L_list):
            storage = DataStorage(file_path=f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/')
            storage.num_h = 400
            positions = storage.read_positions()
            h_list = np.linspace(0.0, 2.0, storage.num_h)
            plt.figure()
            plt.axhline(0, color='grey', linewidth=1, linestyle='--',alpha=0.5)
            _, signs = get_interactions(storage, positions, k=0, threshold1=None, threshold2=None)
            
            for ia, a in enumerate(a_list):
                sum_currents = get_sumcurrents(storage, GAUSSIAN, a/L)
                flow_list = np.dot(sum_currents, signs)
                plt.plot(h_list, flow_list, label=rf'$a={a:.1f}$', color=colors[ia], linewidth=1.5, alpha = 0.7)
            
            plt.xlabel(r'$h$')
            plt.ylabel(r'$I_{\text{circ}}$')
            plt.legend(loc='lower left',frameon=False,labelspacing=0.25)
            ax = plt.gca()
            ax.text(0.02, 0.98, rf"$L={L}$", transform=ax.transAxes, fontsize=10, verticalalignment='top')
            ax.tick_params(direction='in', which='both')
            ax.minorticks_on()
            ax_inset = inset_axes(ax, width="45%", height='35%', loc="upper right")
            hc_plot_exist(ax_inset)
            ax_inset.tick_params(axis='both', labelsize=8)
            
            plt.savefig(f"/Users/ruiqixu/Desktop/{L}.png",dpi=300, bbox_inches='tight')
            #plt.show()
            plt.close()
    
    def storage_info():
        L_list = [9, 13, 17, 21, 25, 29]
        a_list = [0.5, 1.0, 2.0, 4.0]
        diff_area = [[None, 0.4],[0.4, 0.6],[None, 0.6,], [None, None]]
        for iL, L in enumerate(L_list):
            storage = DataStorage(file_path=f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/')
            if L == 29:
                storage.num_h = 200
            else:
                storage.num_h = 400
            # [area, a, num_h]
            big_flow_list = np.zeros((4, 4, storage.num_h))
            positions = storage.read_positions()
            for iarea, area in enumerate(diff_area):
                _, signs = get_interactions(storage, positions, k=0, threshold1=area[0], threshold2=area[1])
                for ia, a in enumerate(a_list):
                    sum_currents = get_sumcurrents(storage, GAUSSIAN, a/L)
                    flow_list = np.dot(sum_currents, signs)
                    big_flow_list[iarea,ia]=flow_list
            np.save(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/big_flow_list_correction.npy', big_flow_list)
    
    def currents_diffsize_two_visualizations_plot_plot():
        L_list = [9, 13, 17, 21, 25, 29]
        a_list = [0.5, 1.0, 2.0, 4.0]
        ia = 2
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, 6)][::-1]
        plt.figure()
        plt.axhline(0, color='grey', linewidth=1, linestyle='--',alpha=0.5)
        for iL, L in enumerate(L_list):
            big_flow_list = np.load(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/big_flow_list.npy')
            if L == 29:
                h_list = np.linspace(0.0, 2.0, 200)
            else:
                h_list = np.linspace(0.0, 2.0, 400)
            plt.plot(h_list, big_flow_list[3, ia]/(2/L), label=rf'$L={L}$', color=colors[iL], linewidth=1.5, alpha = 0.7)
        plt.xlabel(r'$h$')
        plt.ylabel(r'$I_{\text{circ}}$')
        plt.xticks(np.arange(0, 2.1, 0.1))
        #plt.ylim(-0.08, 0.08)
        plt.legend(loc='upper center',frameon=False,labelspacing=0.7)
        ax = plt.gca()
        ax.text(0.02, 0.98, rf"$a={a_list[ia]}$", transform=ax.transAxes, fontsize=10, verticalalignment='top')
        ax.tick_params(direction='in', which='both')
        ax.minorticks_on()
        ax.vlines(0.9, -0.08, 0, linestyles='--', colors='black',linewidth=1,alpha=0.5)
        ax.plot(0.9, -0.08, 'o', markerfacecolor='black', markeredgecolor='black', markersize=2,clip_on=False)
        
        ax_inset = inset_axes(ax, width="45%", height="45%",loc='upper right',bbox_to_anchor=(0, -0.01, 1, 1),bbox_transform=ax.transAxes)
        current_Jr(km.DEFECT, km.SINGLE, ax_inset, 1.3)
        ax_inset.axis("off")
        
        ax_inset = inset_axes(ax, width="45%", height="45%",loc='lower left',bbox_to_anchor=(0, 0.01, 1, 1),bbox_transform=ax.transAxes)
        current_Jr(km.DEFECT, km.SINGLE, ax_inset, 0.7)
        ax_inset.axis("off")

        ax.annotate(
            '',
            xy=(0.67, -0.073),
            xytext=(0.7, -0.08),
            arrowprops=dict(arrowstyle="->", color='black'))
        ax.plot(0.7, -0.08, 'o', markerfacecolor='black', markeredgecolor='black', markersize=2,clip_on=False)
        ax.annotate(
            '',
            xy=(1.33, -0.073),
            xytext=(1.3, -0.08),
            arrowprops=dict(arrowstyle="->", color='black'))
        ax.plot(1.3, -0.08, 'o', markerfacecolor='black', markeredgecolor='black', markersize=2,clip_on=False)
        
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        plt.show()
        #plt.savefig(f"/Users/ruiqi/Documents/tmp/currents/fig1b.png",dpi=300, bbox_inches='tight')
        
    def new_cancel_out_plot():
        h_list = np.linspace(0.0, 2.0, 400)
        big_flow_list = np.load('/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/25/big_flow_list1.npy')
        my_legends = [['forestgreen', 'NNN', '--'],['forestgreen', 'NN', ':'],
                          ['forestgreen','NNN+NN', '-'], ['royalblue', 'total system', '-']]
        plt.figure()
        plt.axhline(0, color='grey', linewidth=1, linestyle=':',alpha=0.4)
        for i, my_legend in enumerate(my_legends):
            plt.plot(h_list, big_flow_list[i,2], linestyle=my_legend[2], color=my_legend[0], label=my_legend[1])
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        y_max = max(abs(y_min), abs(y_max))
        ax.set_ylim(-y_max, y_max)
        ax.text(0.02, 0.98, rf"$L={25},\;a={2.0}$", transform=ax.transAxes, fontsize=10, verticalalignment='top')
        ax.tick_params(direction='in', which='both')
        ax.minorticks_on()
        plt.xlabel(r'$h$')
        plt.ylabel(r'$I_\text{circ}$')
        plt.legend(frameon=False,loc='lower left')

        ax_inset = inset_axes(ax, width="35%", height="35%",loc='upper right')
        image = mpimg.imread("/Users/ruiqi/Documents/tmp/currents/fig4b_sub.png")
        ax_inset.imshow(image)
        ax_inset.axis("off")
        plt.savefig(f"/Users/ruiqi/Documents/tmp/currents/fig4b.png",dpi=300, bbox_inches='tight')
        #plt.show()
            
    def cancel_out_plot():
        L_list = [25]#[9, 13, 17, 21, 25, 29]
        a = 1
        for iL, L in enumerate(L_list):
            storage = DataStorage(file_path=f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/')
            storage.num_h = 400
            positions = storage.read_positions()
            h_list = np.linspace(0.0, 2.0, storage.num_h)
            
            various_set = [[[None, 0.4, 'green', 'inner NNN'],[0.4, 0.6,'dodgerblue', 'inner NN'],[None, 0.6, 'crimson','inner']],
                    [[None, 0.6, 'crimson', 'inner'],[0.6, None, 'mediumpurple', 'outer'],[None, None, 'orange', 'total']]] #第一组star hex in 第二组in out total
            my_linestyle={"0":"-", "10":":"}
            folder_name = ['star_hex_in', 'in_out_all']
            for ig, group in enumerate(various_set):
                plt.figure()
                plt.axhline(0, color='grey', linewidth=1, linestyle='--',alpha=0.5)
                for area in group:
                    for k in [0, 10]:
                        _, signs = get_interactions(storage, positions, k=k, threshold1=area[0], threshold2=area[1])
                        sum_currents = get_sumcurrents(storage, GAUSSIAN, a/L)
                        flow_list = np.dot(sum_currents, signs)
                        plt.plot(h_list, flow_list, linestyle=my_linestyle[f"{k}"], color=area[2])
                ax = plt.gca()
                y_min, y_max = ax.get_ylim()
                y_max = max(abs(y_min), abs(y_max))
                ax.set_ylim(-y_max, y_max)
                ax.text(0.02, 0.98, rf"$L={L},\;a={a}$", transform=ax.transAxes, fontsize=10, verticalalignment='top')
                ax.tick_params(direction='in', which='both')
                ax.minorticks_on()
                plt.xlabel(r'$h$')
                plt.ylabel(r'$I_\text{circ}$')
                custom_legend = [
                    Line2D([0], [0], color=group[0][2], lw=2, label=group[0][3]),
                    Line2D([0], [0], color=group[1][2], lw=2, label=group[1][3]),
                    Line2D([0], [0], color=group[2][2], lw=2, label=group[2][3])
                ]
                plt.legend(handles=custom_legend, frameon=False,loc='lower left')
                plt.tight_layout()
                plt.savefig(f"/Users/ruiqixu/Desktop/{ig}.png",dpi=300, bbox_inches='tight')
                #plt.show()
                plt.close()

    def currents_diffsize_correction():
        L_list = [9, 13, 17, 21, 25, 29]
        a_list = [0.5, 1.0, 2.0, 4.0]
        ia = 2
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, 6)][::-1]
        plt.figure()
        plt.axhline(0, color='grey', linewidth=1, linestyle='--',alpha=0.5)
        for iL, L in enumerate(L_list):
            big_flow_list = np.load(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/big_flow_list_correction.npy')
            if L == 29:
                h_list = np.linspace(0.0, 2.0, 200)
            else:
                h_list = np.linspace(0.0, 2.0, 400)
            plt.plot(h_list, big_flow_list[3, ia], label=rf'$L={L}$', color=colors[iL], linewidth=1.5, alpha = 0.7)
        plt.xlabel(r'$h$')
        plt.ylabel(r'$I_{\text{circ}}$')
        plt.xticks(np.arange(0, 2.1, 0.1))
        #plt.ylim(-0.08, 0.08)
        plt.legend(loc='upper center',frameon=False,labelspacing=0.7)
        ax = plt.gca()
        ax.text(0.02, 0.98, rf"$a={a_list[ia]}$", transform=ax.transAxes, fontsize=10, verticalalignment='top')
        ax.tick_params(direction='in', which='both')
        ax.minorticks_on()
        # ax.vlines(0.9, -0.08, 0, linestyles='--', colors='black',linewidth=1,alpha=0.5)
        # ax.plot(0.9, -0.08, 'o', markerfacecolor='black', markeredgecolor='black', markersize=2,clip_on=False)
        
        # ax_inset = inset_axes(ax, width="45%", height="45%",loc='upper right',bbox_to_anchor=(0, -0.01, 1, 1),bbox_transform=ax.transAxes)
        # current_Jr(km.DEFECT, km.SINGLE, ax_inset, 1.3)
        # ax_inset.axis("off")
        
        # ax_inset = inset_axes(ax, width="45%", height="45%",loc='lower left',bbox_to_anchor=(0, 0.01, 1, 1),bbox_transform=ax.transAxes)
        # current_Jr(km.DEFECT, km.SINGLE, ax_inset, 0.7)
        # ax_inset.axis("off")

        # ax.annotate(
        #     '',
        #     xy=(0.67, -0.073),
        #     xytext=(0.7, -0.08),
        #     arrowprops=dict(arrowstyle="->", color='black'))
        # ax.plot(0.7, -0.08, 'o', markerfacecolor='black', markeredgecolor='black', markersize=2,clip_on=False)
        # ax.annotate(
        #     '',
        #     xy=(1.33, -0.073),
        #     xytext=(1.3, -0.08),
        #     arrowprops=dict(arrowstyle="->", color='black'))
        # ax.plot(1.3, -0.08, 'o', markerfacecolor='black', markeredgecolor='black', markersize=2,clip_on=False)
        
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        #plt.show()
        plt.savefig(f"/Users/ruiqi/Documents/tmp/currents/fig1b_correction.png",dpi=300, bbox_inches='tight')

    def new_cancel_out_plot_correction():
        h_list = np.linspace(0.0, 2.0, 400)
        big_flow_list = np.load('/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/25/big_flow_list_correction.npy')
        my_legends = [['forestgreen', 'NNN', '--'],['forestgreen', 'NN', ':'],
                          ['forestgreen','NNN+NN', '-'], ['royalblue', 'total system', '-']]
        plt.figure()
        plt.axhline(0, color='grey', linewidth=1, linestyle=':',alpha=0.4)
        for i, my_legend in enumerate(my_legends):
            plt.plot(h_list, big_flow_list[i,2], linestyle=my_legend[2], color=my_legend[0], label=my_legend[1])
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        y_max = max(abs(y_min), abs(y_max))
        ax.set_ylim(-y_max, y_max)
        ax.text(0.02, 0.98, rf"$L={25},\;a={2.0}$", transform=ax.transAxes, fontsize=10, verticalalignment='top')
        ax.tick_params(direction='in', which='both')
        ax.minorticks_on()
        plt.xlabel(r'$h$')
        plt.ylabel(r'$I_\text{circ}$')
        plt.legend(frameon=False,loc='lower left')

        # ax_inset = inset_axes(ax, width="35%", height="35%",loc='upper right')
        # image = mpimg.imread("/Users/ruiqi/Documents/tmp/currents/fig4b_sub.png")
        # ax_inset.imshow(image)
        # ax_inset.axis("off")
        plt.savefig(f"/Users/ruiqi/Documents/tmp/currents/fig4b_correction.png",dpi=300, bbox_inches='tight')
        #plt.show()

    #all_plot()
    #cancel_out_plot()
    #new_cancel_out_plot()
    storage_info()
    #currents_diffsize_two_visualizations_plot_plot()
    #currents_diffsize_correction()
    #new_cancel_out_plot_correction()

import sympy as sp
def one_hex_model():
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]
    h = sp.symbols('h',real=True,nonnegative=True)
    #order: E C A D F B
    H = sp.Matrix([[0, h*sp.I, -h*sp.I, -1, -1, 0], [ -h*sp.I, 0, h*sp.I, -1, 0, -1], [h*sp.I, -h*sp.I, 0, 0, -1, -1],
                   [-1, -1, 0, 0, -h*sp.I, h*sp.I], [-1, 0, -1, h*sp.I, 0, -h*sp.I], [0, -1, -1, -h*sp.I, h*sp.I, 0]])
    eigenvectors = H.eigenvects()
    #print(eigenvectors)
    num_h=200
    h_list = np.linspace(0.0, 2.0, num_h)
    
    fig, ax = plt.subplots()
    thetas = [r'$0$', r'$\pi$', r'$\pi/3$', r'$4\pi/3$', r'$5\pi/3$', r'$2\pi/3$']
    for i in range(6):
        energy_list = np.array([eigenvectors[i][0].subs(h, h_val) for h_val in h_list])
        ax.plot(h_list, energy_list, label=rf"${sp.latex(eigenvectors[i][0])}$" + "\n" + f"{thetas[i]}", color=colors[i])
        # phase_list = np.zeros(200)
        # for j, h_val in enumerate(h_list):
        #     phase_A=sp.arg(eigenvectors[i][2][0][2].subs(h, h_val))
        #     phase_B=sp.arg(eigenvectors[i][2][0][5].subs(h, h_val))
        #     phase_list[j] = (phase_A-phase_B)%(2*pi)
        # ax.plot(h_list[mask], phase_list[mask], color=line.get_color())
    ax.legend(loc="lower right", fontsize=11, bbox_to_anchor=(1.28, 0.0),labelspacing=0.8, frameon=False)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in')
    ax.set_xlabel(r'$h$')
    ax.set_ylabel('Energy')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"/Users/ruiqixu/Desktop/toy.png",dpi=fig.dpi, bbox_inches='tight')
    plt.close()
    
def middle_hex():
    def if_same_pos(pos1, pos2):
        return np.allclose(pos1[0], pos2[0], atol=tolerance) and np.allclose(pos1[1], pos2[1], atol=tolerance)
    
    def phase_info(eigenstate):
        theta_a = np.angle(eigenstate[hex_index[0]])
        theta_b = np.angle(eigenstate[hex_index[1]])
        theta_c = np.angle(eigenstate[hex_index[2]])
        theta_ab = (theta_a - theta_b) % (2 * np.pi)
        theta_bc = (theta_b - theta_c) % (2 * np.pi)
        return theta_ab, theta_bc
    
    def current_info(current):
        current_ba = current[current_index[0]]
        current_ca = current[current_index[1]]
        return current_ba, current_ca
    
    km.change_model(km.DEFECT, km.SINGLE)
    L=km.model['L']=km.model['W']=25
    tmp_sys = km.model_builder()
    tolerance=0.1
    hex_pos = [[0.5, -0.5/sqrt(3)], [0.5, 0.5/sqrt(3)], [0, 1/sqrt(3)], [-0.5, 0.5/sqrt(3)], [-0.5, -0.5/sqrt(3)], [0, -1/sqrt(3)]]
    hex_index = np.zeros(6, dtype=int)
    current_pos = [[1, 0], [2, 0], [2, 1], [3, 1]]
    current_index = np.zeros(4, dtype=int)
    for i, site in enumerate(tmp_sys.sites):
        for which_pos, pos in enumerate(hex_pos):
            if if_same_pos(site.pos, pos):
                hex_index[which_pos]=i
    for i, (tail, head) in enumerate(tmp_sys.graph):
        for which_current, pos in enumerate(current_pos):
            if if_same_pos(tmp_sys.sites[head].pos, hex_pos[pos[1]]) and if_same_pos(tmp_sys.sites[tail].pos, hex_pos[pos[0]]):
                current_index[which_current]=i
    #print(hex_index)
    #print(current_index)
    
    for h in [0.5, 1, 1.5]:
        km.model['h'] = h
        sys = km.model_builder()
        J = kwant.operator.Current(sys)
        H = sys.hamiltonian_submatrix(sparse=False)
        evals, evecs = eigh(H)
        condition = (evals < 0) & (np.abs(evals) < 0.8)
        selected_indices = np.where(condition)[0]
        selected_evals = evals[selected_indices]
        selected_evecs = evecs[:, selected_indices]
        sorted_order = np.argsort(np.abs(selected_evals))
        sorted_evals = selected_evals[sorted_order]
        sorted_evecs = selected_evecs[:, sorted_order]
        currents = np.array([J(sorted_evecs[:, i]) for i in range(len(sorted_evals))])

        y_values = np.array([phase_info(sorted_evecs[:, i]) for i in range(len(sorted_evals))])
        x_axis = sorted_evals
        plt.scatter(x_axis, y_values[:, 0], color='blue', alpha=0.6, s=30, label=r'$\theta_A-\theta_B$',edgecolors='none')
        plt.scatter(x_axis, y_values[:, 1], color='orange', alpha=0.6, s=30, label=r'$\theta_B-\theta_C$',edgecolors='none')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.085), ncol=2, frameon=False)
        plt.title(rf"$h={h}$", loc='left')
        plt.xlabel('Energy')
        ax = plt.gca()
        ax.tick_params(direction='in')
        #plt.xlim(-4, 0)#for L=1
        plt.xlim(-0.82, 0.02)#for L=25
        plt.ylabel(r'$\theta$')
        plt.ylim(-0.15, 2 * np.pi+0.15)
        plt.xticks(fontsize=12)
        interval = np.pi/3
        plt.yticks([i * interval for i in range(7)], [r'$0$', r'$\pi/3$', r'$2\pi/3$',r"$\pi$", r'$4\pi/3$', r'$5\pi/3$', r'$2π$'],fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        #plt.show()
        plt.savefig(f"/Users/ruiqixu/Desktop/{h}_phase.png",dpi=300, bbox_inches='tight')
        plt.close()
        
        y_values = np.array([current_info(currents[i]) for i in range(len(sorted_evals))])
        x_axis = sorted_evals
        plt.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.4)
        plt.scatter(x_axis, y_values[:, 0], color='blue', alpha=0.6, s=30, label=r'$J_{B\leftarrow A}$',edgecolors='none')
        plt.scatter(x_axis, y_values[:, 1], color='hotpink', alpha=0.6, s=30, label=r'$J_{C\leftarrow A}$',edgecolors='none')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.085), ncol=2, frameon=False)
        plt.title(rf"$h={h}$", loc='left')
        plt.xlabel('Energy')
        ax = plt.gca()
        ax.tick_params(direction='in')
        #plt.xlim(-4, 0)#for L=1
        plt.xlim(-0.82, 0.02)#for L=25
        #plt.ylim(-0.32, 0.52)#for L=1
        plt.ylim(-0.02, 0.035)#for L=25
        plt.ylabel(r'$J$')
        #plt.ylim(-0.03, 0.05)
        plt.xticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        #plt.show()
        plt.savefig(f"/Users/ruiqixu/Desktop/{h}_current.png",dpi=300, bbox_inches='tight')
        plt.close()

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
    for tail, head in temp_sys.graph:
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

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
#write_data()
read_data()

#test_triangle()
#middle_hex()
#one_hex_model()

def draw_distance_with_letter():
    km.change_model(km.DEFECT, km.SINGLE)
    km.model['L']=km.model['W']=9
    sys = km.model_builder()
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.5, 0.5, 0.5, 0.3),hop_lw=0.07)
    eps = 0.08
    bounds = [(-2.5/sqrt(3)-eps, 2.5/sqrt(3)+eps), (-2.5/sqrt(3), 2.5/sqrt(3))]
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    text_positions = [
    (-1, sqrt(3)/2, r'$d$'),
    (0, sqrt(3)/2, r'$c$'),
    (-1.5, 0, r'$e$'),
    (-0.5, 0.0, r'$b$'),
    (0, -0.5/sqrt(3), r'$a$')]
    for x, y, text in text_positions:
        ax.scatter(x, y, color='black', s=50, clip_on=False)
        ax.text(x+0.1, y, text, fontsize=25, ha='left', va='center')
    plt.savefig(f"/Users/ruiqi/Documents/tmp/currents/r.png",dpi=300, bbox_inches='tight')
    #plt.show()
    
def draw_distance_with_color():
    km.change_model(km.DEFECT, km.SINGLE)
    km.model['L']=km.model['W']=9
    sys = km.model_builder()
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    kwant.plot(sys, ax=ax, show=False, site_color=(0.6, 0.7, 1.0, 0.0), hop_color=(0.5, 0.5, 0.5, 0.3),hop_lw=0.07)
    eps = 0.08
    bounds = [(-2.5/sqrt(3)-eps, 2.5/sqrt(3)+eps), (-2.5/sqrt(3), 2.5/sqrt(3))]
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    dot_positions = [
    (-1, sqrt(3)/2, 'grey'),
    (0, sqrt(3)/2, 'crimson'),
    (-1.5, 0, 'forestgreen'),
    (-0.5, 0.0, 'black'),
    (0, -0.5/sqrt(3), 'royalblue')]
    for x, y, color in dot_positions:
        ax.scatter(x, y, color=color, s=50, clip_on=False)
    plt.savefig(f"/Users/ruiqi/Documents/tmp/currents/r.png",dpi=300, bbox_inches='tight')
    #plt.show()

#draw_distance_with_letter()
#draw_distance_with_color()

#current_Jr(km.DEFECT, km.SINGLE)