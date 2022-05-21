# Rocket Simulation with Explicit Euler's method
# ----------------------------------------------
# 
# @from: https://github.com/casadi/casadi/blob/master/docs/examples/python/rocket.py
#
# System Dynamics
# 
#   sdot = v,
#   vdot = (u - 0.05 * v^2)/m
#   mdot = -0.1*u^2
# 
# Parameters
# 
#   m - mass (kg)
#   s - position (m)
#   v - velocity (m/s)
#   u - fuel/thrust
# 

from casadi import (
    MX,
    Function,
    nlpsol,
    vertcat,
    mtimes,
    linspace,
)
from pylab import (
    plot,
    grid,
    show,
    legend,
    xlabel,
)

# Constants
T = 0.2     # time horizon (s)
N = 20      # euler control interval
dt = T/N    # Time step

# Control
u = MX.sym("u")

# State
x = MX.sym("x",3)
s = x[0] # position
v = x[1] # speed
m = x[2] # mass

# ODE right hand side
sdot = v
vdot = (u - 0.05 * v**2)/m
mdot = -0.1*u**2
xdot = vertcat(sdot, vdot, mdot)

# ODE right hand side function
f = Function('f', [x,u],[xdot])

# Integrate with Explicit Euler over 0.2 seconds
xj = x
for j in range(N):
  fj = f(xj,u)
  xj += dt*fj

# Discrete time dynamics function
F = Function('F', [x,u],[xj])

# Number of control segments
nu = 50 

# Control for all segments
U = MX.sym("U", nu)
 
# Initial conditions
X0 = MX([0, 0, 1])

# Integrate over all intervals
X=X0
for k in range(nu):
  X = F(X, U[k])

# Objective function and constraints
J = mtimes(U.T,U) # u'*u in Matlab
G = X[0:2]     # x(1:2) in Matlab

# NLP
nlp = {'x':U, 'f':J, 'g':G}
 
# Allocate an NLP solver
opts = {"ipopt.tol":1e-10, "expand":True}
solver = nlpsol("solver", "ipopt", nlp, opts)
arg = {}

# Bounds on u and initial condition
arg["lbx"] = -0.5
arg["ubx"] =  0.5
arg["x0"] =   0.4

# Bounds on g
arg["lbg"] = [10,0]
arg["ubg"] = [10,0]

# Solve the problem
res = solver(**arg)

# Get the solution
tgrid = linspace(0, T, nu).full()
plot(tgrid, res["x"].full())
plot(tgrid, res["lam_x"].full())
legend(['u', 'lambda_u'])
xlabel('t (s)')
grid()
show()