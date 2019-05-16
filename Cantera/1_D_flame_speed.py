#!/usr/bin/env python
# coding: utf-8

# # Flame Speed

# In this example we simulate a freely-propagating, adiabatic, 1-D flame and
# * Calculate its laminar burning velocity
# * Perform a sensitivity analysis of its kinetics
# 
#  a flame-fixed co-ordinate system. 
# The reactants enter with density $\rho_{u}$, temperature $T_{u}$ and speed $S_{u}$. 
# The products exit the flame at speed $S_{b}$, density $\rho_{b}$ and temperature $T_{b}$.


# ### Import Modules

# In[ ]:


import cantera as ct
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['lines.linewidth'] = 2

print(f"Running Cantera version {ct.__version__}")


# ### Define the reactant conditions, gas mixture and kinetic mechanism associated with the gas

# In[ ]:


# Define the gas mixture and kinetics
# In this case, we are choosing a modified version of GRI 3.0
gas = ct.Solution('input-files/gri30_noNOx.cti')


# ### Define flame simulation conditions

# In[ ]:


# Inlet temperature in kelvin and inlet pressure in pascal
To = 300
Po = 101325

# Domain width in meters
width = 0.02

# Set the inlet mixture to be stoichiometric CH4 and air 
gas.set_equivalence_ratio(1.0, 'CH4', {'O2':1.0, 'N2':3.76})
gas.TP = To, Po

# Create the flame object
flame = ct.FreeFlame(gas, width=width)

# Set options for the solver
flame.transport_model = 'Mix'
flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)

# Define logging level
loglevel = 1


# In[ ]:


# ### Solve

# The `auto` option in the solve function tries to "automatically" solve the flame 
# =============================================================================
# by applying a few common techniques. 
# First, the flame is solved on a sparse grid with the transport calculations set to mixture averaged. 
# Then grid refinement is enabled, with points added 
# according to the values of the `ratio`, `slope`, and `curve` parameters in the `set_refine_criteria` function. 
# If the initial solve on the sparse grid fails to converge, the simulation is attempted again, 
# but this time with the energy equation disabled. 
# Once the simulation has been solved on the refined grid with the mixture averaged transport, 
# Cantera enables the multicomponent transport and Soret diffusion, if they have been set by the user.
# =============================================================================
# 
# In general, it is recommended that you use the `auto` option the first time you run the solver, 
# =============================================================================
# unless the simulation fails. 
# On subsequent invocations of `solve`, you should not include the `auto` option (or set it to `False`).
# =============================================================================



flame.solve(loglevel=loglevel, auto=True)


# In[ ]:


Su0 = flame.u[0]
print("Flame Speed is: {:.2f} cm/s".format(Su0*100))
flame.show_stats()


# ### Plot figures
# 
# Check and see if all has gone well. Plot temperature and species fractions to see. 
# =============================================================================
# We expect that the solution at the boundaries of the domain will have zero gradient 
# (in other words, that the domain width that we specified is wide enough for the flame).
# =============================================================================


# In[ ]:
# #### Temperature Plot

f, ax = plt.subplots(1, 1)
ax.plot(flame.grid*100, flame.T)
ax.set(xlabel='Distance (cm)', ylabel='Temperature (K)');
# note domain size is not what we originally set -- 
# automatically expanded to satisfy boundary conditions



# In[ ]:
# #### Major species' plot



profile = ct.SolutionArray(gas, shape=len(flame.grid), extra={'z': flame.grid*100})
profile.TPY = flame.T, flame.P, flame.Y.T

f, ax = plt.subplots(1, 1)
ax.plot(profile.z, profile('CH4').X, label=r'CH$_4$')
ax.plot(profile.z, profile('O2').X, label=r'O$_2$')
ax.plot(profile.z, profile('CO2').X, label=r'CO$_2$')
plt.plot(profile.z, profile('H2O').X, label=r'H$_2$O')
ax.legend()
ax.set(xlabel='Distance (cm)', ylabel='Mole fraction');


# ## Sensitivity analysis
# Compute normalized sensitivities of flame speed $S_u$ to changes in the rate coefficient $k_i$ for each reaction
# $$s_i = \frac{k_i}{S_u} \frac{d S_u}{d k_i} $$

# In[ ]:


sens = flame.get_flame_speed_reaction_sensitivities()
# note: much slower for multicomponent / Soret


# In[ ]:


# Find the most important reactions:
sens_data = [(sens[i], gas.reaction_equation(i)) for i in range(gas.n_reactions)]
sens_data.sort(key=lambda item: abs(item[0]), reverse=True)
for s, eq in sens_data[:20]:
    print(f'{s: .2e}  {eq}')


# ## Solving multiple flames (parameter sweep) 

# In[ ]:


# Start  at one limit of the equivalence ratio range
gas.set_equivalence_ratio(0.6, 'CH4', {'O2':1.0, 'N2':3.76})
gas.TP = To, Po

flame = ct.FreeFlame(gas, width=width)

# Enabling pruning is important to avoid continuous increase in grid size
flame.set_refine_criteria(ratio=3, slope=0.15, curve=0.15, prune=0.1)
flame.solve(loglevel=0, refine_grid=True, auto=True)


# In[ ]:


phis = np.linspace(0.6, 1.8, 50)
Su = []

for phi in phis:
    gas.set_equivalence_ratio(phi, 'CH4', {'O2':1.0, 'N2':3.76})
    flame.inlet.Y = gas.Y
    flame.solve(loglevel=0)
    print(f'phi = {phi:.3f}: Su = {flame.u[0]*100:5.2f} cm/s, N = {len(flame.grid)}')
    Su.append(flame.u[0])


# In[ ]:


f, ax = plt.subplots(1, 1)
ax.plot(phis, Su)
ax.set(xlabel='equivalence ratio', ylabel='flame speed (m/s)');


# In[ ]:




