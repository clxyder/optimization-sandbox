from casadi import (
    SX,
    sin, cos,
    Function,
    rootfinder
)

# Define constants
nz = 1
nx = 1

# Define variables
z = SX.sym('x',nz)
x = SX.sym('x',nx)

# Define system of equations
g0 = sin(x+z)
g1 = cos(x-z)
g = Function('g',[z,x],[g0,g1])

# Create solver
G = rootfinder('G','newton',g)
print(G)

# Solve problem
x0 = 1
z0 = 3
r = G(z0,x0)

print(r)