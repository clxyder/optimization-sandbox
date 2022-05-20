from casadi import (
    SX, integrator, cos
)

x = SX.sym('x')
z = SX.sym('z')
p = SX.sym('p')

dae = {
    'x':x,
    'z':z,
    'p':p,
    'ode':z+p,
    'alg':z*cos(z)-x
}

F = integrator('F', 'idas', dae)
print(F)

# Integating DAE from
#   t0=0 - start time
#   tf=1 - end time
# with guess
#   z0=0 - initial guess

x0 = 0
z0 = 0
p0 = 0.2

r = F(x0=x0, z0=z0, p=p0)
print(r['xf'])