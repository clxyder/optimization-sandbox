from casadi import DM, conic

# Create qp matrices
H = 2*DM.eye(2)
A = DM.ones(1,2)
g = DM.zeros(2)
lba = 10.0

# Solver options
qp = {}
qp['h'] = H.sparsity()
qp['a'] = A.sparsity()

# Create solver
S = conic('S', 'qpoases', qp)
print(S)

# Solve problem
r = S(h=H, g=g, a=A, lba=lba)

x_opt = r['x']
print('x_opt: ', x_opt)
