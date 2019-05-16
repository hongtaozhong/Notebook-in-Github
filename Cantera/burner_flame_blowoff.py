#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cantera as ct
import numpy as np
import matplotlib.pyplot as plt


# # Burner Flame Example
# 
# a burner flame at two equivalence ratios, 
#one with a flame attached to the burner surface, 
# =============================================================================
# one where the flame blows off. 
# We will also demonstrate the ability of the Cantera solver to 
# have arbitrary functions inserted to collect data and control the simulation. 
# First, we define a function to set up the flame with the phi as a parameter.
# 
# =============================================================================

# In[2]:


def setup(phi):
    # parameter values
    p = ct.one_atm  # pressure
    tburner = 300.0  # burner temperature
    mdot = 1.2 # kg/m^2/s
    width = 0.05 # m

    gas = ct.Solution('h2o2.cti')

    # set state to that of the unburned gas at the burner
    gas.set_equivalence_ratio(phi, 'H2', 'O2:1.0, AR:5')
    gas.TP = tburner, p

    # Create the stagnation flow object with a non-reactive surface.
    sim = ct.BurnerFlame(gas=gas, width=width)
    # set the mass flow rate at the inlet
    sim.burner.mdot = mdot
    sim.set_refine_criteria(ratio=3, slope=0.16, curve=0.3, prune=0.1)
    return sim


# =============================================================================
# # Now we create a simulation with the equivalence ratio $\phi = 0.6$ 
# which is above the lean blowoff limit for this flame.
# =============================================================================

# In[3]:


sim = setup(0.6)
sim.solve(loglevel=1, auto=True)


# =============================================================================
# # If we plot the temperature on the grid, 
# we will find that the product temperature 
# approaches the equilibrium temperature downstream, 
# indicating a successfully attached flame.
# =============================================================================

# In[4]:


fig, ax = plt.subplots(1, 1)
ax.plot(sim.grid, sim.T)


# =============================================================================
# # On the other hand, let's set up a flame now that blows off the burner. 
# For this case, we will use the equivalence ratio $\phi = 0.4$.
# =============================================================================

# In[5]:


sim = setup(0.4)
sim.solve(loglevel=1, auto=True)

# In[6]:

# Hmm, we only ended up with 25 points in the flame. 
# =============================================================================
# This is strange, so we can add a `callback` function to 
# investigate the progress of the solver as the simulation progresses. 
# We have two options, one which calls the `callback` function after every steady state solve, 
# while the other calls the `callback` function after every transient solve. 
# In this case, we will use the function
# =============================================================================
# 
# ```
# =============================================================================
# # set_steady_callback(self, f)
# #     Set a callback function to be called after each successful steady-state solve, 
# before regridding. The signature of f is float f(float). 
# The argument passed to f is “0” and the output is ignored.
# =============================================================================
# ```
# 
# So, we need to define a function that takes one argument, 
# =============================================================================
# which will always be the value `0`. 
# On the other hand, we can use global variables within the `callback` function. 
# In this case, we want to plot the temperature profile after every steady solver step, 
# so we can track the progress of the solution.
# =============================================================================



# Use callback function to collect and plot data after each steady solve
fig, ax = plt.subplots(1, 1)

def callback(x):
    """
    Callback function that plots the simulation temperature.
    The argument ``x`` is not used.
    """
    ax.plot(sim.grid, sim.T)
    fig.canvas.draw()
    return 0




sim = setup(0.4)

sim.set_steady_callback(callback)
sim.solve(loglevel=0, auto=False)

