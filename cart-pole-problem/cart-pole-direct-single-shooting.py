'''
@from: http://www.matthewpeterkelly.com/tutorials/trajectoryOptimization/cartPoleCollocation.svg

Solution does not quite match up the figure in `frame1106`

'''
# %%
import matplotlib.pyplot as plt
import numpy as np
from casadi import (
    MX,
    vertcat,
    vcat,
    sin, cos,
    integrator,
    nlpsol,
)

# %%
# Constants
N = 20          # number of control intervals
T = 2           # (s) time horizon
g = 9.81        # (m/s^2) acceleration of gravtity on earth
l = 0.5         # (m) length of the pole
m1 = 1          # (kg) mass of the cart
m2 = 0.3        # (kg) mass at the of the pole
d = 1           # (m) track length
d_max = 2*d     # (m) track length
u_max = 20      # (N)(kg*m/s^2) max actuator force

# %%
# Define state variables
u = MX.sym('u')             # control law
q1 = MX.sym('q1')           # cart horizontal position
q2 = MX.sym('q2')           # pole angle
q1dot = MX.sym('q1dot')     # derivtive of position
q2dot = MX.sym('q2dot')     # derivtive of pole angle

x = vertcat(q1,q2,q1dot,q2dot)

# %%
# Dynamic constraints
q1ddot = (
    l*m2*sin(q2)*(q2dot**2) + u + m2*g*cos(q2)*sin(q2)
)/(
    m1 + m2*(1-cos(q2)**2)
)

q2ddot = -1*(
    l*m2*cos(q2)*sin(q2)*(q2dot**2) + u*cos(q2) + (m1+m2)*g*sin(q2)
)/(
    l*(m1 + m2*(1-cos(q2)**2))
)

xdot = vertcat(q1dot,q2dot,q1ddot, q2ddot)

# Objective function
L = u**2

# %%
# Formulate discrete time dynamics
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

x_test = np.zeros(4)
u_test = 3
Ik = I(x0=x_test, p=u_test)
print(Ik['xf'])
print(Ik['qf'])

# %%
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

# Define control law bounds
lbw = -1*u_max*np.ones([N])
ubw = u_max*np.ones([N])

# %%
# Formulate the NLP
# initial conditions
w0 = np.zeros([N,1])
Xk = MX([0, 0, 0, 0])

for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym(f'Uk_{k}')
    w.append(Uk)
    # Integrate until the end of interval
    Ik = I(x0=Xk, p=Uk)
    # Xk = Ik['xf'] + (k/N)*MX([d, np.pi, 0, 0])
    Xk = Ik['xf']
    J += Ik['qf']
    # Add inequality constaint
    g += [Xk]
    if k < N -1:
        # Define state bounds
        lbg += [-1*d_max, -2*np.pi, -1*np.inf, -1*np.inf]
        ubg += [d_max, 2*np.pi, np.inf, np.inf]
    else:
        # Add final state constraint
        lbg += [d, np.pi, 0, 0]
        ubg += [d, np.pi, 0, 0]

# %%
# Create an NLP solver
problem = {
    'f': J,
    'x': vcat(w),
    'g': vcat(g),
}

solver = nlpsol('solver', 'ipopt', problem)

solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = solution['x']

print(w_opt)

# %%
# Process solution for plotting
u_opt = np.array(w_opt)
x_opt = [np.zeros([4,1])]
for k in range(N):
    Ik = I(x0=x_opt[-1], p=u_opt[k])
    x_opt += [Ik['xf'].full()]

q1_opt = [r[0] for r in x_opt]
q2_opt = [r[1] for r in x_opt]

tgrid = [T/N*k for k in range(N+1)]

# %%
# Plot states and control
plt.figure(num=1, figsize=(8, 6), dpi=80)
plt.clf()

plt.subplot(311)
plt.title('States and Control Law')
plt.plot(tgrid, q1_opt, '-')
plt.legend(['q1'])
plt.grid()

plt.subplot(312)
plt.plot(tgrid, q2_opt, '-')
plt.legend(['q2'])
plt.grid()
xlim = plt.xlim()

plt.subplot(313)
plt.plot(tgrid, np.vstack([np.nan, u_opt]), '-.')
plt.xlabel('t')
plt.xlim(xlim)
plt.legend(['u'])
plt.grid()

plt.show()

# %%
