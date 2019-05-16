"""
@author: zhong 
Modified from Cantera examples
Solve a constant pressure ignition problem where the governing equations are
implemented in Python.

Cantera is used for evaluating thermodynamic properties and kinetic rates while
an external ODE solver is used to integrate the resulting equations. In this
case, the SciPy wrapper for VODE is used, which uses the same variable-order BDF
methods as the Sundials CVODES solver used by Cantera.
"""

import cantera as ct
import numpy as np
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
import time

# In[]
start_time = time.time()
f = open("dTdt.csv", 'w+')


# In[]
class ReactorOde(object):
    def __init__(self, gas):
# Parameters of the ODE system and auxiliary data are stored in the ReactorOde object.
        self.gas = gas
        self.P = gas.P
    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - ( (np.dot(self.gas.partial_molar_enthalpies, wdot)+y[0]**2)   /
                  (rho * self.gas.cp))
        dYdt = wdot * self.gas.molecular_weights / rho
        #print(dTdt, file=f)

        print(round(self.gas.T,10), round(dTdt*rho * self.gas.cp,5), sep=" ", file=f)  
        return np.hstack((dTdt, dYdt))
       # hstack: Join a sequence of arrays along a new axis.

# In[]
gas = ct.Solution('gri30.xml')

# Initial condition
P = 1.01e5
gas.TPX = 1200, P, 'H2:2,O2:1,N2:4'
y0 = np.hstack((gas.T, gas.Y))

# In[]

# Set up objects representing the ODE and the solver
ode = ReactorOde(gas)

solver = scipy.integrate.ode(ode)
solver.set_integrator('vode', method='bdf', with_jacobian=True)
solver.set_initial_value(y0, 0.0)

# Integrate the equations, keeping T(t) and Y(k,t)
t_end = 3e-3
states = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
dt = 1e-5

while solver.successful() and solver.t < t_end:
    solver.integrate(solver.t + dt)
    gas.TPY = solver.y[0], P, solver.y[1:]
    states.append(gas.state, t=solver.t)


try:
# Plot the results
    L1 = plt.plot(states.t, states.T, color='r', label='T', lw=2)
    plt.xlabel('time (s)')
    plt.ylabel('Temperature (K)')
    plt.twinx()
    L2 = plt.plot(states.t, states('OH').Y, label='OH', lw=2)
    plt.ylabel('Mass Fraction')
    plt.legend(L1+L2, [line.get_label() for line in L1+L2], loc='lower right')
    plt.show()
    
# Output data    
    dataframe = pd.DataFrame({'t':states.t,'T':states.T})
    dataframe.to_csv("t_T.csv",index=False,sep=' ')
    
    
except ImportError:
    print('Matplotlib not found. Unable to plot results.')

print("--- %s seconds ---" % (time.time() - start_time))