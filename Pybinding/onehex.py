import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, ceil
import cmath
from pybinding.repository import graphene

cc = 1.0
m = 0.0
    
t1 = 1.0
    
def monolayer_graphene(tc, **kwargs):
    t2 = tc*cmath.exp((1.j)*pi/2.)
    t2c = t2.conjugate()
    
    lat = pb.Lattice(a1=[cc, 0], a2=[cc/2, cc* sqrt(3)/2 ])

    lat.add_sublattices(('a', [cc/2, -cc / (2*sqrt(3))], m),
                        ('b', [cc/2, cc / (2*sqrt(3))], -m))

    lat.register_hopping_energies({
        't1': kwargs.get('t1', t1),
        't2': kwargs.get('t2', t2),
        't2c': kwargs.get('t2c', t2c),
    })

    neighbors_graphene = [[0, 0], [0, -1], [1, -1]]
    for neighbor in neighbors_graphene:
        lat.add_hoppings((neighbor, 'a', 'b', -t1))

    neighbors_imag = [(-1, 1), (0, -1), (1, 0)]
    for neighbor in neighbors_imag:
        lat.add_hoppings((neighbor, 'b', 'b', t2))
        lat.add_hoppings((neighbor, 'a', 'a', t2c))
            
    return lat

def vertex(L, W):
    return (L*cc/2, ((W-1)/2*1.5+1)*cc/sqrt(3))

def rectangle(L, W):
    (x0, y0) = vertex(L, W)
    x, y = x0+0.1, y0+0.1
    return pb.Polygon([[x, y], [x, -y], [-x, -y], [-x, y]])

def ifclose(x, y, layer):
    if layer == 1:
        return (x*x + y*y) < (1/12+(layer-1/2)**2+0.01)
    if layer == 2:
        return (x*x + y*y) < (1/12+(layer-1/2)**2+0.01)

def add_defect(layer):
    @pb.hopping_energy_modifier
    def modify_hopping(energy, x1, y1, x2, y2, hop_id):
        condition = ifclose(x1, y1, layer) & ifclose(x2, y2, layer)
        if hop_id == 't2' or hop_id == 't2c':
            energy[~condition] = 0
        return energy
    return modify_hopping

def model_builder(L, W, tc, n):
    model = pb.Model(
        monolayer_graphene(tc),
        add_defect(n),
        rectangle(L, W),
    )
    return model

def whole_ldos(*args):
    model = model_builder(*args)

    solver = pb.solver.lapack(model)
    ldos = solver.calc_spatial_ldos(energy=0, broadening=0.3)  # eV
    plt.figure()
    ldos.plot(site_radius=(0.2, 0.2))
    pb.pltutils.colorbar(label="LDOS")
    plt.figtext(0.5, 1, f'L={args[0]}_W={args[1]}_tc={args[2]:.2f}.png', ha="center", va="top", fontsize=10, color="black")
    plt.savefig(f'./Pybinding/LDOS1/LDOS_L={args[0]}_W={args[1]}_tc={args[2]:.2f}.png') 
    plt.close()
    
# for tc in [0, 0.1, 0.5, 1.0, 1.3, 1.7, 1.8, 2.0, 4.0, 5.0, 10.0]:
#     whole_ldos(13, 13, tc, 2)

def bulk_sites(model, n):
    pos = model.system.positions
    mask = ((pos.x)**2 + (pos.y)**2) <= (1/12+(n-1/2)**2+0.01)
    filtered_positions_x = pos.x[mask]
    filtered_positions_y = pos.y[mask]
    if n == 1:
        assert(len(filtered_positions_x) == 6)
    elif n == 2:
        assert(len(filtered_positions_x) == 24)
    #print(filtered_positions_x, filtered_positions_y)
    return (filtered_positions_x, filtered_positions_y)

def bulk_change_ldos(L, W, n):
    num_sample = 100
    tc_list = np.linspace(0.8, 1.0, num_sample)
    bulk_list = np.zeros((num_sample))
    
    model = model_builder(L, W, 1, 1)
    plt.figure()
    for n in [1, 2]:
        (x_sites, y_sites) = bulk_sites(model, n)
        bulk_list[:] = 0
        for index, tc in enumerate(tc_list):
            model = model_builder(L, W, tc, 1)
            solver = pb.solver.lapack(model)
            for i, x in np.ndenumerate(x_sites):
                temp = solver.calc_ldos(energies=0, broadening=0.2, position=[x, y_sites[i]])
                bulk_list[index] += temp.data[0]
        plt.plot(tc_list, bulk_list, label=f'n={n}')
        
    # bulk_max = f"{tc_list[np.argmax(bulk_list)]:.3f}"
    # A_derivative_max = f"{tc_list[np.argmax(np.diff(A_list))]:.3f}"
    
    # plt.plot(tc_list, B_list)
    # B_max = f"{tc_list[np.argmax(B_list)]:.3f}"
    # B_derivative_max = f"{tc_list[np.argmax(np.diff(B_list))]:.3f}"
    
    # plt.figtext(0.5, 0.98, 'max: '+bulk_max, ha="center", va="top", fontsize=10, color="blue")
    # plt.figtext(0.5, 0.93, 'derivative max - A: '+A_derivative_max+' B: '+B_derivative_max, ha="center", va="top", fontsize=10, color="blue")
    # plt.savefig(f'./Pybinding/Change_L={L}_W={W}.png')
    plt.legend()
    plt.savefig(f'./Pybinding/bulk_change_L={L}_W={W}_energy=0.png')
    # plt.show()
    plt.close()

bulk_change_ldos(13, 13, 1)

def AB_change_ldos(L, W, n):
    num_sample = 500
    tc_list = np.linspace(0.3, 1.8, num_sample)
    A_list, B_list = np.zeros((num_sample)), np.zeros((num_sample))
    for index, tc in enumerate(tc_list):
        model = model_builder(L, W, tc, n)
        solver = pb.solver.lapack(model)
        A = solver.calc_ldos(energies=0, broadening=0.2, position=[0, 1/sqrt(3)])
        A_list[index] = A.data[0]
        B = solver.calc_ldos(energies=0, broadening=0.2, position=[0, 2/sqrt(3)])
        B_list[index] = B.data[0]
        
    plt.figure()
    plt.plot(tc_list, A_list)
    A_max = f"{tc_list[np.argmax(A_list)]:.3f}"
    A_derivative_max = f"{tc_list[np.argmax(np.diff(A_list))]:.3f}"
    
    plt.plot(tc_list, B_list)
    B_max = f"{tc_list[np.argmax(B_list)]:.3f}"
    B_derivative_max = f"{tc_list[np.argmax(np.diff(B_list))]:.3f}"
    
    plt.figtext(0.5, 0.98, 'max - A: '+A_max+' B: '+B_max, ha="center", va="top", fontsize=10, color="blue")
    plt.figtext(0.5, 0.93, 'derivative max - A: '+A_derivative_max+' B: '+B_derivative_max, ha="center", va="top", fontsize=10, color="blue")
    plt.savefig(f'./Pybinding/AB_change_L={L}_W={W}.png')
    plt.show()
    plt.close()
# AB_change_ldos(13, 13, 1)