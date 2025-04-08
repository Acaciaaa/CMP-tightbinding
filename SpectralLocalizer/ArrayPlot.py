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
import os
import re
import pandas as pd
from math import sqrt, pi, sin, cos, dist, isclose
import warnings
from matplotlib.lines import Line2D
from line_profiler import LineProfiler

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

l_list=[6, 8, 10, 12, 14, 16, 18, 20]
l_num = 8
chern_num = 10
gap_num = 100


def extract_data():
    folder_path = '/Users/ruiqixu/Desktop/tmp/current_new/update/update2'
    pattern = re.compile(r'Lx=(\d+)')
    chern_h_list = np.zeros(chern_num)
    chern_list = np.zeros((l_num, chern_num))
    gap_h_list = np.zeros(gap_num)
    gap_list = np.zeros((l_num, gap_num))

    chern_folder_path = folder_path + '/chern'
    for filename in os.listdir(chern_folder_path):
        match = pattern.search(filename)
        if match:
            l = int(match.group(1))
            l_index = (l//2)-3
        else:
            exit
        file_path = os.path.join(chern_folder_path, filename)
        df = pd.read_excel(file_path, header=None)

        third_column = df.iloc[:, 2].to_numpy()
        last_column = df.iloc[:, -1].to_numpy()
        if l_index == 0:
            chern_h_list=third_column
        chern_list[l_index]=last_column
        
    gap_folder_path = folder_path + '/gap'
    for filename in os.listdir(gap_folder_path):
        match = pattern.search(filename)
        if match:
            l = int(match.group(1))
            l_index = (l//2)-3
        else:
            exit

        file_path = os.path.join(gap_folder_path, filename)
        df = pd.read_excel(file_path, header=None)

        third_column = df.iloc[:, 2].to_numpy()
        last_column = df.iloc[:, -1].to_numpy()
        if l_index == 0:
            gap_h_list=third_column
        gap_list[l_index]=last_column
        
    np.save(f'/Users/ruiqixu/Desktop/tmp/current_new/update/update2/array/chern_h_list.npy', chern_h_list)
    np.save(f'/Users/ruiqixu/Desktop/tmp/current_new/update/update2/array/chern_list.npy', chern_list)
    np.save(f'/Users/ruiqixu/Desktop/tmp/current_new/update/update2/array/gap_h_list.npy', gap_h_list)
    np.save(f'/Users/ruiqixu/Desktop/tmp/current_new/update/update2/array/gap_list.npy', gap_list)
#extract_data()

chern_h_list = np.load(f'/Users/ruiqixu/Desktop/tmp/current_new/update/update2/array/chern_h_list.npy')
chern_list = np.load(f'/Users/ruiqixu/Desktop/tmp/current_new/update/update2/array/chern_list.npy')
gap_h_list = np.load(f'/Users/ruiqixu/Desktop/tmp/current_new/update/update2/array/gap_h_list.npy')
gap_list = np.load(f'/Users/ruiqixu/Desktop/tmp/current_new/update/update2/array/gap_list.npy')
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, l_num)][::-1]

for l_set in [[6, 12, 18], [8, 10, 14, 16, 20]]:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), height_ratios=[1, 1.6])
    ax1.axhline(0, color='grey', linewidth=1, linestyle='--',alpha=0.5)
    
    for l in l_set:
        l_index = l//2-3
        ax1.plot(chern_h_list[:5], chern_list[l_index,:5], label = f'{l}', marker='o', linestyle='-', color=colors[l_index],markersize=4,linewidth=2,alpha=0.8,clip_on=False)
        ax1.plot(chern_h_list[5:8], chern_list[l_index,5:8], marker='o', linestyle='-', color=colors[l_index],markersize=3.5,linewidth=1.5,alpha=0.8,clip_on=False)
        ax2.plot(gap_h_list, gap_list[l_index], label = f'{l}', marker='o', linestyle='-',color=colors[l_index],markersize=2.5,linewidth=1.2,alpha=0.8)
    
    ax1.set_ylabel(r"$C$")
    ax1.set_ylim(-1, 1)
    ax1.tick_params(direction='in',which='both')
    #ax1.minorticks_on()
    ax2.set_ylabel("Gap")
    ax2.set_ylim(0, 0.1)
    ax2.tick_params(direction='in',which='both')
    #ax2.minorticks_on()
    
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.set_xlabel(r"$h$")
    ax2.set_xlim(-0.05, 2.05)
    plt.xticks(np.arange(0, 2.1, 0.1))
    plt.subplots_adjust(hspace=0.1)
    if l_set[0] == 6:
        legend = ax1.legend(loc='center left', title=r"$l=6\mathcal{Z}$",title_fontsize=10,fontsize=8)
        plt.savefig(f"/Users/ruiqixu/Desktop/tmp/current_new/update/update2/fig2b.png",dpi=300, bbox_inches='tight')
    else:
        legend = ax1.legend(loc='center left', title=r"$l\neq6\mathcal{Z}$",title_fontsize=10,fontsize=8)
        plt.savefig(f"/Users/ruiqixu/Desktop/tmp/current_new/update/update2/fig2a.png",dpi=300, bbox_inches='tight')
    #legend.get_frame().set_visible(False)
    #plt.show()
    