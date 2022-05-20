from casadi import SX, vertcat, qpsol

# Creating symbolic variables
x = SX.sym('x')
y = SX.sym('y')

# Solver options
qp = {
    'x':vertcat(x,y),   # x vector
    'f':x**2+y**2,      # objective function
    'g':x+y-10,         # constraints
}

# Creating qp solver
S = qpsol('S', 'qpoases', qp) # w/ qpOASES (distributed with CasADi)
print(S)

# Solve problem
r = S(lbg=0)

x_opt = r['x']
print('x_opt: ', x_opt)