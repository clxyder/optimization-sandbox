from casadi import (
    DaeBuilder,
    integrator
)

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
