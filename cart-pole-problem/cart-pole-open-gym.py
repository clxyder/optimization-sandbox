# System Dynamics from https://sci-hub.se/10.1109/tsmc.1983.6313077

import gym
from casadi import (
    sin, cos, pi,
    vertcat,
    linspace,
    sum2,
    sign,
    Opti,
)
import numpy as np
import matplotlib.pyplot as plt

# Define constants
T = 2               # time horizon for simulation (s)
N = 39              # number of control intervals
h_k = T/(N-1)       # time step size used in discretization
l = 0.5             # length of pole (m)
mc = 1              # mass of the cart (kg)
mp = 0.1            # mass of the pole (kg)
g = -9.81           # acceleration of gravity on Earth (m/s^2)
d = 1               # length of track (m)
dmax = 2*d          # max length of track (m)
umax = 20           # max torque (N)
mu_c = 0.0005       # coefficient of friction of cart on track
mu_p = 0.000002     # coefficient of friction of cart on track
M = 5               # number of decision variables

class CartPoleController:
    '''
    Uses Hermite-Simpson Collocation method to solve cart pole problem
    '''
    def __init__(self) -> None:
        self.setup()

    def setup(self, xk_prev = None, uk_prev = None):
        if xk_prev is None:
            xk_prev = [0,0,0,0]
        if uk_prev is None:
            uk_prev = 0

        self.opti = Opti()
        self.p = self.opti.variable(M,N)

        # Extract variables
        x, u = self.p[:4,:], self.p[4,:]

        q1, q2 = x[0,:], x[1,:]
        q1dot, q2dot = x[2,:], x[3,:]

        # Interpolation constraint symbolic expressions
        xk = x[:,:-1]
        xk1 = x[:,1:]
        uk = u[:-1]
        uk1 = u[1:]
        uc = (uk + uk1)/2       # calculation of u_{k+(1/2)}

        # Objective function
        J = (h_k/6)*sum2(uk**2 + 4*(uc**2) + uk1**2)
        self.opti.minimize(J)

        fk = self.f(xk,uk)           # collocation point at x_k
        fk1 = self.f(xk1,uk1)        # collocation point at x_{k+1}

        # Mid-point collocation point
        xc = 1/2 * (xk + xk1) + h_k/8 * (fk - fk1)  # x_{k+(1/2)}
        fc = self.f(xc,uc)           # collocation point at f_{k+(1/2)}

        # Simpson quadrature constraint in `compressed form`
        G = xk-xk1 + h_k/6*(fk + 4*fc + fk1)
        self.opti.subject_to(G==0)   # Add interpolation constraint

        # Add path constraints
        self.opti.subject_to(q1 < dmax)
        self.opti.subject_to(-dmax < q1)
        self.opti.subject_to(u < umax)
        self.opti.subject_to(-umax < u)

        # Add final boundary constraint
        self.opti.subject_to(x[:,-1]==[d,pi,0,0])

        # Add starting boundary constraint
        self.opti.subject_to(x[:,0] == xk_prev)

        # Update initial conditions with previous state & control
        self.opti.set_initial(q1[:], linspace(xk_prev[0],d,N))
        self.opti.set_initial(q2[:], linspace(xk_prev[1],pi,N))
        self.opti.set_initial(q1dot[:], linspace(xk_prev[2],0,N))
        self.opti.set_initial(q2dot[:], linspace(xk_prev[3],0,N))

        self.opti.set_initial(u[:], uk_prev*np.ones([N]))

        # Set solver properties
        self.opti.solver('ipopt', {}, {'print_level': 0})

    def act(self, state, uk_prev):
        x, x_dot, theta, theta_dot = state
        xk = [x, theta, x_dot, theta_dot]

        self.setup(xk, uk_prev)
        sol = self.opti.solve()

        u = self.p[4,1]
        u_opt = sol.value(u)

        return int(u_opt > 0), sol

    def f(self, x, u):
        q1 = x[0,:]     # cart position (m)
        q2 = x[1,:]     # angle position (rad)
        q1dot = x[2,:]  # cart velocity (m/s)
        q2dot = x[3,:]  # angular velocity (rad/s)

        q2ddot = - (
            g*sin(q2) + cos(q2)*( (-u - mp*l*(q2dot**2)*sin(q2) + mu_c*sign(q1dot))/(mc + mp) ) - ((mu_p*q2dot)/(mp*l))
        ) / (
            l*( (4/3) - (mp*cos(q2)**2)/(mc + mp) )
        )

        q1ddot = (
            u + mp*l*(q2dot**2 * sin(q2) - q2ddot*cos(q2)) - mu_c*sign(q1dot)
        ) / (
            mc + mp
        )
        
        return vertcat(q1dot, q2dot, q1ddot, q2ddot)


if "__main__" == __name__:

    # Initialize environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(state_size)
    print(action_size)

    x, x_dot, theta, theta_dot = env.reset()
    x_init = [x, theta, x_dot, theta_dot]
    u_init = 0

    print(x_init)

    # Initialize controller

    controller = CartPoleController()
    # sol = controller.opti.solve()

    rewards = []

    # first iteration
    controller.setup(x_init, u_init)
    sol = controller.opti.solve()
    uk = sol.value(controller.p[4,:])[1]
    action = int(uk > 0)

    next_state, reward, done, _ = env.step(action)

    # second iteration
    for tk in range(500):
        print("iteration: ", tk)

        x, x_dot, theta, theta_dot = next_state
        xk = [x, theta, x_dot, theta_dot]

        print(xk)

        controller.setup(xk, uk)
        sol = controller.opti.solve()
        uk = sol.value(controller.p[4,:])[1]
        action = int(uk > 0)

        next_state, reward, done, _ = env.step(action)

        rewards.append(reward)

        if done:
            break

    plt.plot(rewards)

    # # Extract variables
    # x, u = controller.p[:4,:], controller.p[4,:]

    # q1, q2 = x[0,:], x[1,:]
    # q1dot, q2dot = x[2,:], x[3,:]

    # # Plot the solution
    # q1_opt = sol.value(q1)
    # q2_opt = sol.value(q2)
    # u_opt = sol.value(u)

    # t = np.linspace(0,T,N)

    # plt.figure(1)

    # # Plot cart position
    # plt.subplot(311)
    # plt.plot(t,q1_opt)
    # plt.legend(['q1'])
    # plt.grid()

    # # Plot pole angle
    # plt.subplot(312)
    # plt.plot(t,q2_opt)
    # plt.legend(['q2'])
    # plt.grid()

    # # Plot control law
    # plt.subplot(313)
    # plt.plot(t,u_opt)
    # plt.legend(['u'])
    # plt.grid()

    # plt.show()

    
