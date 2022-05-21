# Car race along a track
# ----------------------
# An optimal control problem (OCP),
# solved with direct multiple-shooting.
#
# For more information see: http://labs.casadi.org/OCP
# 
# @from: https://github.com/casadi/casadi/blob/master/docs/examples/python/race_car.py
# 
# -------------------------------------------------------------------------
# 
#                     .///////////////////*,                                  
#                   ///////%////////%////////////,                            
#                 //////////////////////////(///(#///,                        
#               ,/////////,            ,///////&/#//////*                     
#              ,////////,                    ,%//////(/////,                  
#             ,////////.                         ,///////////*                
#             ////////,                              /////%////,              
#            ////(///*                                 /////////,             
#           ,////////                                   ,////////.            
#           ////////,                                    ////(///,            
#          ////////,  ,*/////////,,                     *////////             
#        //////////,/////////////////////*,         ,//////&////,             
#     ,/////(////,//////////////////(///////////////////(//////               
#    ,////////////////////,..,*///////////////(%%#//////////,                 
#    ,///////////(//////              ,////////////////*,                     
#      //////////////,                                                        
#          ,,,,,                                                              
# 
# 
# -------------------------------------------------------------------------
# 
# We must control u(t) to ensure the race car stays on the track.
# 
#   System Dynamics:
#       d[p(t)v(t)]/dt=[v(t), u(t)−v(t)]
# 
#   p(t) - position (m)
#   v(t) - velocity (m/s)
#   T - time to finish track
# 
# Encode the the task definition in a continuous-time optimal control problem (OCP)
# 
#   minimize        T
#     x,u
#   subject to
#     dynamic constraints
#       xdot = f(x(t),u(t)), t \in [0,T],
#     boundary condition: start at position 0
#       p(t0) = 0,
#     boundary condition: start with zero speed
#       v(t0) = 0,
#     boundary condition: the finish line is at position 1
#       p(T) = 1,
#     path constraint: throttle is limited
#       0 <= u(t) < 1,
#     path constraint: speed limit varying along the track
#       v(t) <= L(p(t))
# 
# The following is a multiple-shooting transcription of the original OCP
# 
#   minimize        T
#   x_k, u_j, 
#    k \in [1,N+1],
#    j \in [1, N]
# 
#   subject to
#     (1) dynamic constraints a.k.a gap closing
#       x_{k+1} = F(x_k,u_k), k \in [1,N],
#     (2) boundary condition: start at position 0
#       p(t0) = 0,
#     (3) boundary condition: start with zero speed
#       v(t0) = 0,
#     (4) boundary condition: the finish line is at position 1
#       p(T) = 1,
#     (5) path constraint: throttle is limited
#       0 <= u(t) < 1,
#     (6) path constraint: speed limit varying along the track
#       v(t) <= L(p(t))
# 

from casadi import (
    Opti,
    vertcat,
    sin,
    jacobian,
    hessian,
    dot,
    pi,
)

from matplotlib.pyplot import (
    plot,
    step,
    figure,
    legend,
    show,
    spy
)

N = 100 # number of control intervals

opti = Opti() # Optimization problem

# ---- decision variables ---------
X = opti.variable(2,N+1) # state trajectory
pos   = X[0,:]
speed = X[1,:]
U = opti.variable(1,N)   # control trajectory (throttle)
T = opti.variable()      # final time

# ---- objective          ---------
opti.minimize(T) # race in minimal time

# ---- dynamic constraints --------
f = lambda x,u: vertcat(x[1],u-x[1]) # dx/dt = f(x,u)

dt = T/N # length of a control interval
for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   Xk, Uk = X[:,k], U[:,k]
   k1 = f(Xk, Uk)
   k2 = f(Xk+dt/2*k1, Uk)
   k3 = f(Xk+dt/2*k2, Uk)
   k4 = f(Xk+dt*k3, Uk)
   x_next = Xk + dt/6*(k1+2*k2+2*k3+k4) 
   opti.subject_to(X[:,k+1] == x_next) # close the gaps

# ---- path constraints -----------
limit = lambda pos: 1-sin(2*pi*pos)/2
opti.subject_to(speed <= limit(pos))    # track speed limit
opti.subject_to(opti.bounded(0, U, 1))  # control is limited

# ---- boundary conditions --------
opti.subject_to(pos[0] == 0)   # start at position 0
opti.subject_to(speed[0] == 0) # start from stand-still 
opti.subject_to(pos[-1] == 1)  # finish line at position 1

# ---- misc. constraints  ----------
opti.subject_to(T >= 0) # Time must be positive

# ---- initial values for solver ---
opti.set_initial(speed, 1)
opti.set_initial(T, 1)

# ---- solve NLP              ------
opti.solver("ipopt") # set numerical backend
sol = opti.solve()   # actual solve

# ---- post-processing        ------
plot(sol.value(speed),label="speed")
plot(sol.value(pos),label="pos")
plot(limit(sol.value(pos)).full(),'r--',label="speed limit")
step(range(N),sol.value(U),'k',label="throttle")
legend(loc="upper left")

# Uncomment to see jacobian & hessian sparsity
# figure()
# spy(sol.value(jacobian(opti.g,opti.x)))
# figure()
# spy(sol.value(hessian(opti.f+dot(opti.lam_g,opti.g),opti.x)[0]))

show()
