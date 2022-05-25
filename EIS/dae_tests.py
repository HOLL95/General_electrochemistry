from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from scikits.odes import dae


#data of the pendulum
l = 1.0
m = 1.0
g = 1.0
#initial condition
theta0= np.pi/3 #starting angle
x0=np.sin(theta0)
y0=-(l-x0**2)**.5
lambdaval = 0.1
z0  = [x0, y0, 0., 0., lambdaval]
zp0 = [0., 0., lambdaval*x0/m, lambdaval*y0/m-g, -g]
def residual(t, x, xdot, result):
    """ we create the residual equations for the problem"""
    result[0] = x[2]-xdot[0]
    result[1] = x[3]-xdot[1]
    result[2] = -xdot[2]+x[4]*x[0]/m
    result[3] = -xdot[3]+x[4]*x[1]/m-g
    result[4] = x[2]**2 + x[3]**2 \
                    + (x[0]**2 + x[1]**2)/m*x[4] - x[1] * g
    print(result)
solver = dae('ida', residual,
             compute_initcond='yp0',
             first_step_size=1e-18,
             atol=1e-6,
             rtol=1e-6,
             algebraic_vars_idx=[4],
             compute_initcond_t0 = 60,
             old_api=False)
solution = solver.solve([0., 1., 2.], z0, zp0)


print('\n   t        Solution')
print('----------------------')
for t, u in zip(solution.values.t, solution.values.y):
    print('{0:>4.0f} {1:15.6g} '.format(t, u[0]))

#Solve over the next hour by continuation
times = np.linspace(0, 3600, 61)
times[0] = solution.values.t[-1]
solution = solver.solve(times, solution.values.y[-1], solution.values.ydot[-1])
if solution.errors.t:
    print ('Error: ', solution.message, 'Error at time', solution.errors.t)
print ('Computed Solutions:')
print('\n   t        Solution ')
print('-----------------------')
for t, u in zip(solution.values.t, solution.values.y):
    print('{0:>4.0f} {1:15.6g}'.format(t, u[0]))
solver = dae('ida', residual,
             compute_initcond='yp0',
             first_step_size=1e-18,
             atol=1e-6,
             rtol=1e-6,
             algebraic_vars_idx=[4],
             compute_initcond_t0 = 60,
             old_api=False,
             max_steps=5000)
solution = solver.solve(times, solution.values.y[-1], solution.values.ydot[-1])
if solution.errors.t:
    print ('Error: ', solution.message, 'Error at time', solution.errors.t)
print ('Computed Solutions:')
print('\n   t        Solution')
print('----------------------')
for t, u in zip(solution.values.t, solution.values.y):
    print('{0:>4.0f} {1:15.6g} '.format(t, u[0]))
