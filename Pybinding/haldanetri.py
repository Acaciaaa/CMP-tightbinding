import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
from cmath import exp
from pybinding.repository import graphene

cc = 1   
t1 = 1   
tc = 0.5
temp = tc*exp((1.j)*pi/2.)
t2 = 0.2
t3 = 0.3
m = 0
m2 = -0.35

def haldane_triangular(**kwargs):
    lat = pb.Lattice(a1=[cc, 0],
                     a2=[cc/2, cc* sqrt(3)/2 ])
    
    lat.add_sublattices(('c', [0, 0], m2),
                        ('a', [cc/2, -cc / (2*sqrt(3))], m),
                        ('b', [cc/2, cc / (2*sqrt(3))], -m))
    
    # lat.register_hopping_energies({
    #     't1': kwargs.get('t1', t1),
    #     'tc': kwargs.get('tc', tc),
    #     't2c': kwargs.get('t2c', t2c),
    # })

    neighbors_second = [[0, 0], [-1, 1], [-1, 0]]
    for neighbor in neighbors_second:
        lat.add_hoppings((neighbor, 'c', 'a', -t3))
    neighbors_second = [[0, 0], [0, -1], [-1, 0]]
    for neighbor in neighbors_second:
        lat.add_hoppings((neighbor, 'c', 'b', -t3))

    neighbors_graphene = [[0, 0], [0, -1], [1, -1]]
    for neighbor in neighbors_graphene:
        lat.add_hoppings((neighbor, 'a', 'b', -t1))
        
    neighbors_triangular = [(0, 1), (1, 0), (-1, 1)]
    for neighbor in neighbors_triangular:
        lat.add_hoppings((neighbor, 'c', 'c', -t2))
        
    neighbors_imag = [(0, 1), (-1, 0), (1, -1)]
    for neighbor in neighbors_imag:
        lat.add_hoppings((neighbor, 'b', 'b', temp))
        lat.add_hoppings((neighbor, 'a', 'a', temp.conjugate()))
        
    return lat

def vertex(L, W):
    return (L*cc/2, ((W-1)/2*1.5+1)*cc/sqrt(3))
    
def rectangle(L, W):
    (x0, y0) = vertex(L, W)
    x, y = x0+0.1, y0+0.1
    return pb.Polygon([[x, y], [x, -y], [-x, -y], [-x, y]])

model = pb.Model(
    haldane_triangular(),
    # pb.rectangle(10),
    # pb.translational_symmetry(a2=False),
    rectangle(11, 11),
)
fig, ax = plt.subplots()
model.plot()
plt.show()

# solver = pb.solver.lapack(model)
# left = [-2*math.pi / (3*math.sqrt(3)), 2*math.pi / 3]
# K1 = [-4*math.pi / (3*math.sqrt(3)), 0]
# right = [-2*math.pi / (3*math.sqrt(3)), -2*math.pi / 3]

# bands = solver.calc_bands(left, K1, right)
# bands.plot(point_labels=['K\'', 'K', 'K\''])
# plt.show()

# solver = pb.solver.lapack(model)
# bands = solver.calc_bands(0, 2*pi)
# bands.plot()
# plt.show()

solver = pb.solver.lapack(model)
ldos = solver.calc_spatial_ldos(energy=0, broadening=cc/(2*sqrt(3)))  # eV
fig, ax = plt.subplots()
(x0, y0) = vertex(11, 11)
ax.set_xlim([-x0, 0])
ax.set_ylim([-y0, 0])
ldos.plot(site_radius=(0.2, 0.5))
pb.pltutils.colorbar(label="LDOS")
plt.show()
plt.close(fig)
