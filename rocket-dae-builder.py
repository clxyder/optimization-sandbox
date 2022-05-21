from casadi import (
    DaeBuilder,
    integrator,
    hcat,
    vcat,
)

# Constants
N = 21
g = 9.81 # earth's gravity m/s2

dae = DaeBuilder()

# Add input expressions
a = dae.add_p('a')
b = dae.add_p('b')
u = dae.add_u('u')
h = dae.add_x('h')
v = dae.add_x('v')
m = dae.add_x('m')

# Add output expressions
hdot = v
vdot = (u-a*v**2)/m-g
mdot = -b*u**2
dae.add_ode('hdot', hdot)
dae.add_ode('vdot', vdot)
dae.add_ode('mdot', mdot)

# Specify initial conditions
dae.set_start('h', 0)
dae.set_start('v', 0)
dae.set_start('m', 1)

# Add meta information
dae.set_unit('h','m')
dae.set_unit('v','m/s')
dae.set_unit('m','kg')

# Display DAE
dae.disp(True)

### Create ODE solver function
ode_f = dae.create('f', ['x','u','p'], ['ode'])
print("ode_f: ", ode_f)

x0 = [2,3,1]
u0 = 2
p0 = [2,3,u0]

# Solve ODE
r = ode_f(x0, u0, p0)
print("x_dot: ", r)

### Simulate system
# Create integrator
daeSetup = {
    'x': vcat(dae.x),
    'p': vcat([*dae.p, *dae.u]),
    'ode': vcat(dae.ode),
}

I = integrator('F', 'idas', daeSetup)
print(I)

# Create vectors
X = [2,3,1]
u0 = 2
P = [2,3,u0]
J = 0

for k in range(N):
    P[2] = P[2] * (1/(k+1))
    Ik = I(x0=X, p=P)
    X = Ik['xf']
    J += Ik['qf']
    print(f'X: {X}\nP: {P}\nJ: {J}')
