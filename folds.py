from casadi import (
    MX,
    Function
)

'''
iteration depends on the result from the previous iteration, the fold
construct applies. In the following, the `x` variable acts as an accumulater
that is initialized by `x0 ∈ R^n`
'''

# Constants
N = 6

# Variables
x0 = MX.sym('x')

# Define function
f = Function('f', [x0], [13*x0])
print(f)

# The following are equivalent

# `x` acts as an accumlator
x = x0
for _ in range(N):
    x = f(x)
F = Function('F', [x0], [x])
print(F)

# fold construct
F = f.fold(N)
print(F)
