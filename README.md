# Optimization Sandbox

I am on a journey to learn optimization, during college I took courses in Numerical Optimization and Optimal Control, needless to say I was fascinated by the subjects.

The plan is to learn enough material to use in a career for development of space vehicles.

I will focus on:

* Trajectory Optimization
* Model Predictive Control
* Monte Carlo Simulations

I plan to use [CasADi](https://web.casadi.org/) to faciliate my learning experience. This repository will contain example code from their [examaple repo](https://github.com/casadi/casadi/tree/master/docs/examples/python) and other ideas I come up with.

## Trajectory Optimization

### Transcription Methods

* Indirect Methods
  * "optimize then discretize"
  * more accurate
  * harder to pose and solve
* Direct Methods
  * "discretize and optimize"
  * less accurate
  * easier to pose and solve

### Trajectory Methods

* Shooting Methods
  * Based on simulation
  * Better for problems with simple control and no path constraints
* Collocation Methods
  * Based on function approximation
  * Better for problems with complicated control and/or path constraints

### Fine Tunning

* h-methods
  * Low-order method
  * Converge by increasing number of segments
  * Increaseing the number of control units increases accuracy
* p-methods
  * High-order methods
  * Converge by increasing method order
  * Increasing method order provide better fitting, thus increasing accuracy
* Adaptive Methods
  * Uses novel methods to switch between `h-methods` and `p-methods` to improve results

### Trajectory Optimization Methods

* Collocation Methods
  * Trapezoidal direct collocation
  * Hermite–Simpson direct collocation
  * Global orthogonal collocation (Chebyshev Lobatto)
* Shooting Methods
  * Direct Single shooting
  * Direct multiple shooting (4th-order Runge–Kutta)

## Resources

* [Trajectory Optimization Tutorial: Slides](http://www.matthewpeterkelly.com/tutorials/trajectoryOptimization/cartPoleCollocation.svg)
* [An Introduction to Trajectory Optimization: How to do your Direct Collocation](https://www.matthewpeterkelly.com/research/MatthewKelly_IntroTrajectoryOptimization_SIAM_Review_2017.pdf)
* [Transcription Methods for Trajectory Optimization](https://arxiv.org/pdf/1707.00284.pdf)
