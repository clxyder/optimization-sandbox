from casadi import SX, vertcat, nlpsol

# Creating symbolic variables
x: SX.sym = SX.sym('x')
y: SX.sym = SX.sym('y')
z: SX.sym = SX.sym('z')

# Solver options
nlp = {
    'x':vertcat(x,y,z), # x vector
    'f':x**2+100*z**2,  # objective function
    'g':z+(1-x)**2-y,   # constraints
}

# Create solver
S = nlpsol('S', 'ipopt', nlp)
print(S)

# Initial condition
x_init = [2.5,3.0,0.75]

# Solve problem
r = S(x0=x_init,lbg=0, ubg=0)

# Extract solution
x_opt = r['x']
print('x_opt: ', x_opt)