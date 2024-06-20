import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
import math
from pybinding.repository import graphene

def monolayer_graphene(**kwargs):
    a = math.sqrt(3)   # [nm] unit cell length
    a_cc = 1  # [nm] carbon-carbon distance
    t1 = -1      # [eV] nearest neighbour hopping
    
    t2 = 1*np.exp((1.j)*np.pi/2.)
    t2c= t2.conjugate()

    lat = pb.Lattice(a1=[a, 0],
                     a2=[a/2, a/2 * math.sqrt(3)])
    lat.add_sublattices(('A', [0, -a_cc/2], 0),
                        ('B', [0,  a_cc/2], 0))
    
    lat.register_hopping_energies({
        't1': kwargs.get('t1', t1),
        't2': kwargs.get('t2', t2),
        't2c': kwargs.get('t2c', t2c),
    })
    lat.add_hoppings(
        ([0,  0], 'A', 'B', 't1'),
        ([1, -1], 'A', 'B', 't1'),
        ([0, -1], 'A', 'B', 't1')
    )
    lat.add_hoppings(
        ([1,  0], 'A', 'A', 't2'),
        ([0, -1], 'A', 'A', 't2'),
        ([-1, 1], 'A', 'A', 't2'),
        ([1,  0], 'B', 'B', 't2c'),
        ([0, -1], 'B', 'B', 't2c'),
        ([-1, 1], 'B', 'B', 't2c')
    )
    return lat

epsilon = 0.02
def ifclose(n, target):
    return np.abs(n - target) < epsilon

def add_one_haldane():
    @pb.hopping_energy_modifier
    def modify_hopping(energy, x1, y1, x2, y2, hop_id):
        # print('x1', x1, 'y1', y1, hop_id)
        # print('energy', energy)

        if hop_id == 't2':
            condition1 = ifclose(x1, 0) & ifclose(y1, -0.5) & ifclose(x2, math.sqrt(3)) & ifclose(y2, -0.5)
            condition2 = ifclose(x1, math.sqrt(3)) & ifclose(y1, -0.5) & ifclose(x2, math.sqrt(3)/2) & ifclose(y2, 1)
            condition3 = ifclose(x1, math.sqrt(3)/2) & ifclose(y1, 1) & ifclose(x2, 0) & ifclose(y2, -0.5)
            energy[~condition1 & ~condition2 & ~condition3] = 0
        if hop_id == 't2c':
            condition1 = ifclose(x1, 0) & ifclose(y1, 0.5) & ifclose(x2, math.sqrt(3)) & ifclose(y2, 0.5)
            condition2 = ifclose(x1, math.sqrt(3)) & ifclose(y1, 0.5) & ifclose(x2, math.sqrt(3)/2) & ifclose(y2, -1)
            condition3 = ifclose(x1, math.sqrt(3)/2) & ifclose(y1, -1) & ifclose(x2, 0) & ifclose(y2, 0.5)
            energy[~condition1 & ~condition2 & ~condition3] = 0
            
        # print('energy', energy)
        return energy
    return modify_hopping

model = pb.Model(
    monolayer_graphene(),
    #add_one_haldane(),
    pb.primitive(a1=20),
    pb.translational_symmetry(a1=False)
)
# fig, ax = plt.subplots()
# model.plot()
# plt.show()

# solver = pb.solver.lapack(model)
# left = [-2*math.pi / (3*math.sqrt(3)), 2*math.pi / 3]
# K1 = [-4*math.pi / (3*math.sqrt(3)), 0]
# right = [-2*math.pi / (3*math.sqrt(3)), -2*math.pi / 3]

# bands = solver.calc_bands(left, K1, right)
# bands.plot(point_labels=['K\'', 'K', 'K\''])
# plt.show()

solver = pb.solver.lapack(model)
ldos = solver.calc_spatial_ldos(energy=0, broadening=0.2)  # eV
fig, ax = plt.subplots()
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ldos.plot(site_radius=(0.2, 0.5))
pb.pltutils.colorbar(label="LDOS")
plt.savefig('LDOS2.png') 
plt.close(fig)
