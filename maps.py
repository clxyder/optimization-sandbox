import numpy as np
from casadi import SX, MX, hcat, Function, mtimes, vcat

n = 2
m = 4
N = 6

x = MX.sym('y', n, 1) # represents the [2,1] vector in the for-loop
X = MX.sym("X", n, N) # the entire X matrix

m = np.vstack((np.eye(2), np.ones([1,2]))) # the transformation matrix

f = Function('f', [x], [mtimes(m,x)])
print(f)

ys = []
for i in range(N):
  ys.append(f(X[:,i]))

Y = hcat(ys)
F = Function('F',[X],[Y])
print(F)

X1 = np.vstack(([2,3,4,5,6,7],
                [3,4,5,6,7,8]))
print(X1)

print(F(X1))

# map construct
F = f.map(N)
print(F)

# threadmap construct
F = f.map(N,"thread",2)
print(F)