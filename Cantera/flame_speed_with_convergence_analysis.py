#!/usr/bin/env python
# coding: utf-8

# # Flame Speed with Convergence Analysis

# In this example we simulate a freely-propagating, adiabatic, 1-D flame and
# * Calculate its laminar burning velocity
# * Estimate the uncertainty in the laminar burning velocity calculation, due to grid size.
# 
# The figure below illustrates the setup, in a flame-fixed co-ordinate system. 
# The reactants enter with density $\rho_{u}$, temperature $T_{u}$ and speed $S_{u}$. 
# The products exit the flame at speed $S_{b}$, density $\rho_{b}$ and temperature $T_{b}$.

# * Discretization error affects flame speed calculation
# * Discretization of the convection term is first order
# * Expect the error to be proportional to $\Delta x$, or inversely proportional to the number of grid points ($N$)
#   $$S_{u,\mathrm{observed}} = S_{u,\mathrm{true}} + \frac{k}{N}$$
# * We can calculate the flame speed on multiple grid sizes, then extrapolate as $N\rightarrow\infty$
# * We can estimate the error both with respect to the extrapolated flame speed, and the error in the extrapolation
# * Use 1D solver "callback" functions to analyze the grid after each refinement step

# ### Import Modules

# In[ ]:


import cantera as ct
import numpy as np
import scipy.optimize
import time
from IPython.display import display, HTML

from matplotlib import pyplot as plt


# In[ ]:


plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
print(f"Running Cantera version {ct.__version__}")


# ## Estimate uncertainty from grid size and speeds

# In[ ]:


def extrapolate_uncertainty(grids, speeds, plot=True):
    """
    Given a list of grid sizes and a corresponding list of flame speeds,
    extrapolate and estimate the uncertainty in the final flame speed.
    Also makes a plot.
    """
    grids = list(grids)
    speeds = list(speeds)
    def speed_from_grid_size(grid_size, true_speed, error):
        """
        Given a grid size (or an array or list of grid sizes), return a prediction
        (or array of predictions) of the computed flame speed, based on the parameters
        `true_speed` and `error`        
        """
        return true_speed +  error / np.array(grid_size)

    # Use last 4 grids to extrapolate
    popt, pcov = scipy.optimize.curve_fit(speed_from_grid_size, grids[-4:], speeds[-4:])
    true_speed_estimate = popt[0]
    perr = np.sqrt(np.diag(pcov))

    percent_error_in_true_speed = 100 * perr[0] / true_speed_estimate
    estimated_percent_error = 100 * (speed_from_grid_size(grids[-1], *popt) - true_speed_estimate) / true_speed_estimate
    total_percent_error_estimate = abs(percent_error_in_true_speed) + abs(estimated_percent_error)

    print(f"Fitted true_speed is {popt[0]*100:.4f} ± {perr[0]*100:.4f} cm/s ({percent_error_in_true_speed:.2f}%)")
    print(f"Difference from extrapolated speed {estimated_percent_error:.2f}%")
    print(f"Estimated total error {total_percent_error_estimate:.2f}%")
    
    if not plot:
        return true_speed_estimate, total_percent_error_estimate

    f, ax = plt.subplots(1, 1)
    ax.semilogx(grids, speeds,'o-')
    ax.set_ylim(min(speeds[-5:] + [true_speed_estimate-perr[0]]) * 0.95,
                max(speeds[-5:]+[true_speed_estimate+perr[0]]) * 1.05)
    ax.plot(grids[-4:], speeds[-4:], 'or')
    extrapolated_grids = np.logspace(np.log10(grids[0]), np.log10(8 * grids[-1]), 100)
    ax.plot(extrapolated_grids, speed_from_grid_size(extrapolated_grids, *popt),':r')
    xlims = min(extrapolated_grids), max(extrapolated_grids)
    ax.set_xlim(xlims)
    ax.hlines(true_speed_estimate, *xlims, colors='r', linestyles='dashed')

    ax.hlines([true_speed_estimate-perr[0], true_speed_estimate+perr[0]], *xlims,
              colors='r', linestyles='dashed', alpha=0.3)
    ax.fill_between(xlims, true_speed_estimate-perr[0],true_speed_estimate+perr[0], facecolor='red', alpha=0.1)

    above = popt[1] / abs(popt[1]) # will be +1 if approach from above or -1 if approach from below
    
    local_speed_estimate = speed_from_grid_size(grids[-1], *popt)
    local_err = np.array([[max(true_speed_estimate-local_speed_estimate, 0)],
                          [max(local_speed_estimate-true_speed_estimate, 0)]])
    extrap_err = np.array([[abs(perr[0])], [0]])
    ax.errorbar([grids[-1]], [true_speed_estimate], local_err, capsize=10, color='k', lw=2, capthick=2)
    ax.errorbar([grids[-1]*3], [true_speed_estimate], abs(perr[0]), capsize=10, color='k', lw=2, capthick=2)
        
    ax.annotate(f"{abs(estimated_percent_error):.2f}%",
                xy=(grids[-1], local_speed_estimate),
                xycoords='data',
                xytext=(10,20*above),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->',
                               connectionstyle='arc3')
                )

    ax.annotate(rf"$\pm${abs(percent_error_in_true_speed):.2f}%",
                xy=(grids[-1]*3, true_speed_estimate-abs(perr[0])),
                xycoords='data',
                xytext=(10,-20*above),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->',
                                connectionstyle='arc3')
                )

    ax.set(ylabel="Flame speed (m/s)", xlabel="Grid size")
    plt.show()
    
    return true_speed_estimate, total_percent_error_estimate


# In[ ]:


def make_callback(flame):
    speeds = []
    grids = []

    def callback(_):
        speed = flame.u[0]
        grid = len(flame.grid)
        speeds.append(speed)
        grids.append(grid)
        print(f"Iteration {len(grids)} ({grids[-1]} points). Current flame speed is is {speed*100:.4f} cm/s")
        if len(grids) < 5:
            return 1.0 # 
        try:
            extrapolate_uncertainty(grids, speeds)
        except Exception as e:
            print("Couldn't estimate uncertainty. " + str(e))
            return 1.0 # continue anyway

        return 1.0
    return callback, speeds, grids


# ### Define the gas mixture and kinetic mechanism

# In[ ]:


gas = ct.Solution('input-files/gri30_noNOx.cti')


# ### Set flame simulation conditions and solver properties

# In[ ]:


# Temperature and pressure
To = 300
Po = ct.one_atm

# Domain width in meters
width = 0.06

# Set the gas to be a stoichiometric CH4/air mixture 
gas.set_equivalence_ratio(0.9, 'CH4', {'O2':1.0, 'N2':3.76})
gas.TP = To, Po

# Disable logging
loglevel = 0

# Create the flame object
flame = ct.FreeFlame(gas, width=width)

# Define tight tolerances for the solver
flame.set_refine_criteria(ratio=2, slope=0.01, curve=0.01)

# Set maxiumum number of grid points to be very high (otherwise default is 1000)
flame.set_max_grid_points(flame.flame, 10000)

# Set up the the callback function and lists of speeds and grids
callback, speeds, grids = make_callback(flame)
flame.set_steady_callback(callback)


# ### Solve

# In[ ]:


flame.solve(loglevel=loglevel, auto=True)


# In[ ]:


best_true_speed_estimate, best_total_percent_error_estimate =  extrapolate_uncertainty(grids, speeds)


# In[ ]:


def analyze_errors(grids, speeds):
    true_speed_estimates = []
    total_percent_error_estimates = []
    actual_extrapolated_percent_errors = []
    actual_raw_percent_errors = []
    for i in range(3, len(grids)):
        print(f"{grids[i]} point grid:")
        true_speed_estimate, total_percent_error_estimate = extrapolate_uncertainty(grids[:i+1], speeds[:i+1], plot=False)
        actual_extrapolated_percent_error = 100 * abs(true_speed_estimate - best_true_speed_estimate) / best_true_speed_estimate
        actual_raw_percent_error = 100 * abs(speeds[i] - best_true_speed_estimate) / best_true_speed_estimate
        print(f"Actual extrapolated error (with hindsight) {actual_extrapolated_percent_error:.1f}%")
        print(f"Actual raw error (with hindsight) {actual_raw_percent_error:.1f}%")

        true_speed_estimates.append(true_speed_estimate)
        total_percent_error_estimates.append(total_percent_error_estimate)
        actual_extrapolated_percent_errors.append(actual_extrapolated_percent_error)
        actual_raw_percent_errors.append(actual_raw_percent_error)
        print()

    f, ax = plt.subplots()
    ax.loglog(grids[3:], actual_raw_percent_errors,'o-', label='raw error')
    ax.loglog(grids[3:], actual_extrapolated_percent_errors,'o-', label='extrapolated error')
    ax.loglog(grids[3:], total_percent_error_estimates,'o-', label='estimated error')
    ax.set(ylabel="Error in flame speed (%)", xlabel="Grid size", title=flame.get_refine_criteria())
    ax.yaxis.set_major_formatter
    ax.legend()


# In[ ]:


analyze_errors(grids, speeds)


# ## Middling refine criteria

# In[ ]:


refine_criteria = {'ratio':3, 'slope': 0.1, 'curve': 0.1}


# In[ ]:


# Reset the gas
gas.set_equivalence_ratio(0.9, 'CH4', {'O2':1.0, 'N2':3.76})
gas.TP = To, Po

# Create a new flame object
flame = ct.FreeFlame(gas, width=width)

flame.set_refine_criteria(**refine_criteria)
flame.set_max_grid_points(flame.flame, 1e4)

callback, speeds, grids = make_callback(flame)
flame.set_steady_callback(callback)

flame.solve(loglevel=loglevel, auto=True)


# In[ ]:


analyze_errors(grids, speeds)


# In[ ]:




