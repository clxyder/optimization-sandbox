import numpy as np
from casadi import SX, Function, sin

x = SX.sym('x',2)
y = SX.sym('y')
f = Function('f',[x,y],\
           [x,sin(y)*x])
print(f)

print(x.shape)
print(y.shape)
# print(f.shape) # has no shape object

x1 = np.ones([2,1])
y1 = 3

r = f(x1, y1)
print(r)