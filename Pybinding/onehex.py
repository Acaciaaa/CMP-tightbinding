import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
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

epsilon = 0.7
def ifclose(n, target):
    return np.abs(n - target) < epsilon

def add_one_haldane():
    @pb.hopping_energy_modifier
    def modify_hopping(energy, x1, y1, x2, y2, hop_id):
        condition = ifclose(x1, 0) & ifclose(y1, 0) & ifclose(x2, 0) & ifclose(y2, 0)
        if hop_id == 't2' or hop_id == 't2c':
            energy[~condition] = 0
        return energy
    return modify_hopping

def model_builder(L, W, tc):
    model = pb.Model(
        monolayer_graphene(tc),
        add_one_haldane(),
        rectangle(L, W),
    )
    return model

def whole_ldos(L, W, tc):
    model = model_builder(L, W, tc)

    solver = pb.solver.lapack(model)
    ldos = solver.calc_spatial_ldos(energy=0, broadening=0.2)  # eV
    plt.figure()
    ldos.plot(site_radius=(0.2, 0.2))
    pb.pltutils.colorbar(label="LDOS")
    plt.figtext(0.5, 1, f'L={L}_W={W}_tc={tc:.2f}.png', ha="center", va="top", fontsize=10, color="black")
    plt.savefig(f'./Pybinding/LDOS/LDOS_L={L}_W={W}_tc={tc:.2f}.png') 
    plt.close()
    
# for tc in [0, 0.1, 0.4, 0.6, 0.8, 1, 1.1, 1.3, 3]:
#     whole_ldos(13, 13, tc)

def change_ldos(L, W):
    num_sample = 500
    tc_list = np.linspace(0.3, 1.8, num_sample)
    A_list, B_list = np.zeros((num_sample)), np.zeros((num_sample))
    for index, tc in enumerate(tc_list):
        model = model_builder(L, W, tc)
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
    plt.savefig(f'./Pybinding/Change_L={L}_W={W}.png')
    plt.show()
    plt.close()
        
change_ldos(14, 13)