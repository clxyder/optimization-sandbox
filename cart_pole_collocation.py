'''
@from: https://github.com/justinberi/casadi/blob/ed884e48f3588aba55be8ee675924b292b29b449/docs/examples/python/cart_pole_collocation.py

Solution matches up nicely the figure in:

http://www.matthewpeterkelly.com/tutorials/trajectoryOptimization/cartPoleCollocation.svg#frame1106

'''

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

T = 2
N = 39
h_k = T/(N-1)
l = 0.5
m1 = 1
m2 = 0.3
g = 9.81
d = 1
dmax = 2*d
umax = 20
M = 5

def f(x,u):
    q1 = x[0,:]
    q2 = x[1,:]
    q1dot = x[2,:]
    q2dot = x[3,:]
    q1ddot = (
        l * m2 * ca.sin(q2) * ca.power(q2dot,2) + u + m2 * g * ca.cos(q2) * ca.sin(q2)
    ) / (
        m1 + m2 * (1 - ca.power(ca.cos(q2),2))
    )
    q2ddot = - (
        l * m2 * ca.cos(q2) * ca.sin(q2) * ca.power(q2dot,2) + u * ca.cos(q2) + (m1 + m2) * g * ca.sin(q2)
    ) / (
        l * m1 + l * m2 * (1 - ca.power(ca.cos(q2),2))
    )
    return ca.vertcat(q1dot, q2dot, q1ddot, q2ddot)


# Set up the optimization problem
opti = ca.Opti()

p = opti.variable(M,N)

# Extract variables
x = p[:4,:]
u = p[4,:]

q1 = x[0,:]
q2 = x[1,:]
q1dot = x[2,:]
q2dot = x[3,:]

# Objective function
J = h_k/2*ca.sum2(ca.power(u[:-1],2) + ca.power(u[1:],2))
opti.minimize(J)

# Interpolation constraint symbolic expressions
xk = x[:,:-1]
xk1 = x[:,1:]
uk = u[:-1]
uk1 = u[1:]

fk = f(xk,uk)
fk1 = f(xk1,uk1)
uc = (uk + uk1)/2
xc = 1/2 * (xk + xk1) + h_k/8 * (fk - fk1)
fc = f(xc,uc)

G = xk-xk1 + h_k/6*(fk + 4*fc + fk1)
opti.subject_to(G==0) # Add interpolation constraint


# Add path constraints
opti.subject_to(q1 < dmax)
opti.subject_to(-dmax < q1)
opti.subject_to(u < umax)
opti.subject_to(-umax < u)

# Add boundary constraints
opti.subject_to(x[:,0]==[0,0,0,0])
opti.subject_to(x[:,-1]==[d,ca.pi,0,0])

# Intialize
opti.set_initial(q1[:],ca.linspace(0,d,N))
opti.set_initial(q2[:],ca.linspace(0,ca.pi,N))
opti.set_initial(q1dot[:],ca.linspace(0,0,N))
opti.set_initial(q2dot[:],ca.linspace(0,0,N))
opti.set_initial(u[:],ca.linspace(0,0,N))

opti.solver('ipopt')
sol = opti.solve()

# Plot the solution
q1_opt = sol.value(q1)
q2_opt = sol.value(q2)
u_opt = sol.value(u)

t = np.linspace(0,T,N)

plt.figure(1)

# Plot cart position
plt.subplot(311)
plt.plot(t,q1_opt)
plt.legend(['q1'])
plt.grid()

# Plot pole angle
plt.subplot(312)
plt.plot(t,q2_opt)
plt.legend(['q2'])
plt.grid()

# Plot control law
plt.subplot(313)
plt.plot(t,u_opt)
plt.legend(['u'])
plt.grid()

plt.show()
