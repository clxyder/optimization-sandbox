'''
@from: https://github.com/casadi/casadi/blob/master/docs/examples/python/direct_single_shooting.py
'''

import matplotlib.pyplot as plt
import numpy as np
from casadi import (
    MX,
    vcat,
    integrator,
    nlpsol,
)

# Define constants
T = 10 # time horizon
N = 20 # number of control intervals

# Declare model variables
x0 = MX.sym('x0')
x1 = MX.sym('x1')
x = vcat([x0, x1])
u = MX.sym('u')

# Dynamic constraints
x0_dot = (1-x1**2)*x0 - x1 + u
x1_dot = x0
xdot = vcat([x0_dot, x1_dot])

# Objective function
L = x0**2 + x1**2 + u**2

# Formulate discrete time dynamics. The integrator function
# will first discretize the ODE into a DAE. Then, using the
# `cvodes` interface we can integrate the values to extract
# the future `xf` state and quadrature `qf`.
dae = {
    'x': x,
    'p': u,
    'ode': xdot,
    'quad': L,
}
opts = {
    'tf': T/N,
}

I = integrator('I', 'cvodes', dae, opts)

# Evaluate at a test point
x_test = [0.2, 0.3]
u_test = 0.4
Ik = I(x0=x_test, p=u_test)

print(Ik['xf'])
print(Ik['qf'])

# Start with an empty NLP

'''
Non-linear Program Problem statement:

minimize:           f (x,p)
   x 
subject to: x_lb <=   x    <= x_ub
            g_lb <= g(x,p) <= g_ub

'''

w = []      # vector of u_k at each control interval
w0 = []     # initial condition for said vector
lbw = []    # lower bounds for w
ubw = []    # upper bounds for w
J = 0       # accumulated quadrature
g=[]        # vector of variable subject to inequality constraint
lbg = []    # lower bounds for g
ubg = []    # upper bounds for g

# Formulate the NLP
lbw = -1*np.ones([N,1])
ubw = np.ones([N,1])
w0 = np.zeros([N,1])

lbg = -0.25*np.ones([N,1])
ubg = np.inf * np.ones([N,1])

Xk = MX([0, 1])

for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym(f'Uk_{k}')
    w.append(Uk)
    # Integrate until the end of interval
    Ik = I(x0=Xk, p=Uk)
    Xk = Ik['xf']
    J += Ik['qf']
    # Add inequality constaint
    g.append(Xk[0])

# Create an NLP solver
problem = {
    'f': J,
    'x': vcat(w),
    'g': vcat(g),
}

solver = nlpsol('solver', 'ipopt', problem)

# Solve problem
solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = solution['x']

# Process solution for plotting
u_opt = np.array(w_opt)
x_opt = [[np.array(0), np.array(1)]]
for k in range(N):
    Ik = I(x0=x_opt[-1], p=u_opt[k])
    x_opt += [Ik['xf'].full()]
x0_opt = [r[0] for r in x_opt]
x1_opt = [r[1] for r in x_opt]

tgrid = [T/N*k for k in range(N+1)]

# Plot states and control
plt.figure(1)
plt.clf()
plt.title('States and Control Law')
plt.plot(tgrid, x0_opt, '--')
plt.plot(tgrid, x1_opt, '-')
plt.step(tgrid, np.vstack([np.nan, u_opt]), '-.')
plt.xlabel('t')
plt.legend(['x0', 'x1', 'u'])
plt.grid()
plt.show()
