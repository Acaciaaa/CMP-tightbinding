import numpy as np
import pandas as pd

def generate_csv(*args):
    if args[0] == '1b':
        L_list = [9, 13, 17, 21, 25, 29, 33, 37]
        ia = 2
        df1 = pd.DataFrame()
        h_list = np.linspace(0.0, 2.0, 400)
        df1['h'] = h_list
        df2 = pd.DataFrame()
        h_list = np.linspace(0.0, 2.0, 200)
        df2['h'] = h_list
        for iL, L in enumerate(L_list):
            big_flow_list = np.load(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/{L}/big_flow_list_correction.npy')
            if L >= 29:
                df2[f'{L}'] = big_flow_list[3, ia]
            else:
                df1[f'{L}'] = big_flow_list[3, ia]
        df1.to_csv(f"/Users/ruiqi/Documents/tmp/currents/plot data/fig1b/smaller.csv", index=False)
        df2.to_csv(f"/Users/ruiqi/Documents/tmp/currents/plot data/fig1b/larger.csv", index=False)

    elif args[0] == '2':
        chern_h_list = np.load(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/array/chern_h_list.npy')
        chern_list = np.load(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/array/chern_list.npy')
        gap_h_list = np.load(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/array/gap_h_list.npy')
        gap_list = np.load(f'/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/array/gap_list.npy')
        df1 = pd.DataFrame()
        df1['h'] = chern_h_list
        for l_set in [[6, 12, 18], [8, 10, 14, 16, 20]]:
            for l in l_set:
                l_index = l//2-3
                df1[f'{l}'] = chern_list[l_index]
            
        df1.to_csv(f"/Users/ruiqi/Documents/tmp/currents/plot data/fig2/chern.csv", index=False)
        df2 = pd.DataFrame()
        df2['h'] = gap_h_list
        for l_set in [[6, 12, 18], [8, 10, 14, 16, 20]]:
            for l in l_set:
                l_index = l//2-3
                df2[f'{l}'] = gap_list[l_index]
        df2.to_csv(f"/Users/ruiqi/Documents/tmp/currents/plot data/fig2/gap.csv", index=False)
    
    elif args[0] == '3':
        L_list = [9, 13, 17, 21, 25, 29, 33, 37]
        df = pd.DataFrame()
        h_list = np.linspace(0.0, 2.0, 400)
        df['h'] = h_list
        for iL, L in enumerate(L_list):
            df[f'{L}'] = args[1][iL]
        df.to_csv(f"/Users/ruiqi/Documents/tmp/currents/plot data/fig3/M.csv", index=False)
    
    elif args[0] == '4a':
        r = np.load(f'/Users/ruiqi/Documents/tmp/currents/RKYY/r_0.7.npy')
        currents1 = np.load(f'/Users/ruiqi/Documents/tmp/currents/RKYY/currents_0.7.npy')
        currents2 = np.load(f'/Users/ruiqi/Documents/tmp/currents/RKYY/currents_1.3.npy')
        df = pd.DataFrame({
            'r': r,
            '0.7': currents1,
            '1.3': currents2
        })
        df.to_csv(f"/Users/ruiqi/Documents/tmp/currents/plot data/fig4a/I_circ.csv", index=False)

    elif args[0] == '4b':
        h_list = np.linspace(0.0, 2.0, 400)
        df = pd.DataFrame()
        df['h'] = h_list
        big_flow_list = np.load('/Users/ruiqi/GaTech Dropbox/Ruiqi Xu/data/single/25/big_flow_list_correction.npy')
        names = ['NNN','NN', 'NNN+NN','total system', ]
        for i, name in enumerate(names):
            df[name] = big_flow_list[i,2]
        df.to_csv(f"/Users/ruiqi/Documents/tmp/currents/plot data/fig4b/I_circ.csv", index=False)


for i in ['2']:#, '2', '4a', '4b']:
    generate_csv(i)
# kx = 4 * np.pi / 3
# def curl_phi(x):
#     r = np.abs(x)
#     r = np.where(r == 0, np.nan, r)
#     term1 = 3 * np.cos(kx * x) / r**4
#     term2 = kx * np.sign(x) * np.sin(kx * x) / r**3
#     return term1 + term2
# x = np.linspace(0.5, 6, 550)
# y = curl_phi(x)
# df = pd.DataFrame({
#     'r': x,
#     'RKYY': y
# })
# df.to_csv("/Users/ruiqi/Desktop/RKYY.csv", index=False)