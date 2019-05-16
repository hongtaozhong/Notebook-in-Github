#!/usr/bin/env python
# coding: utf-8

# # 0-D Reactors

# In[1]:


import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

print(f"Using Cantera version {ct.__version__}")


# Cantera can solve the coupled energy and species equations for zero-dimensional reactors. 
# For further details of the actual equations that are solved, 
# see the documentation: https://cantera.org/science/reactors.html 
# The 0-D reactors have a number of options that enable them to model a wide range of systems, 
# including user-controlled varying volume, 
#           heat transfer into or out of the environment, 
#           mass flow into and out of the system, 
#           and surface kinetics.
# 
# Effectively, there are two major choices the user must make in desiging the model of the basic system. 

# The first is whether the reactor solves the general energy equation 
# =============================================================================
# (solved in terms of the mixture enthalpy) 
# or the ideal gas energy equation (solved in terms of the temperature).
# 
#  In general, the two methods will give the same solution to the problem;
#  however, if the system can be modeled as an ideal gas, 
#  the ideal gas equation tends to be more robust and faster.
# # 
# =============================================================================
# The second choice the user must make is whether the system will be modeled as constant pressure or constant volume.
# 
# For each of these choices, there is a specific reactor type:
# 
# * `Reactor`: Constant volume, general energy equation
# * `IdealGasReactor`: Constant volume, ideal gas energy equation
# * `ConstantPressureReactor`: Constant pressure, general energy equation
# * `ConstantPressureIdealGasReactor`: Constant pressure, ideal gas energy equation

# 
# Old examples recommended that 

# =============================================================================
# constant pressure reactors be approximated 
# by allowing the volume of the reactor to expand very quickly as the reaction proceeded. 
# This is no longer recommended because it creates extra stiffness in the solution 
# and the `ConstantPressure*` reactors are a better solution
# =============================================================================

# 
# In the following example, we will solve a constant volume, adiabatic reactor problem, 

#followed by a variable-volume reactor where the volume is controlled by a user-defined function.

# ## RCM Experimental Information

# The following experimental data comes from the work of 
# =============================================================================
# Dames et al., *Combustion and Flame*, 168, 310-330, https://doi.org/10.1016/j.combustflame.2016.02.021 
# The experiment is a 50%-50% by mole mixture of propane and dimethyl ether. 
# The experiment is a rapid compression machine experiment, 
# where a compression process brings the reactant mixture to the end of compression (EOC) conditions. 
# In this particular experiment, the EOC pressure is 30.11 bar and the EOC temperature is 682 K, 
# and the ignition delay is 19.58 ms.
# =============================================================================

# In[2]:


data = np.loadtxt('input-files/0d_example/Tc_682K_P0_1.5137_T0_323K_pressure.txt')
expt_time = data[:, 0]
expt_pres = data[:, 1]
plt.figure('Experimental Pressure')
plt.plot(expt_time, expt_pres)
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]');


# ## Constant Volume Simulation Setup

# In[4]:


gas = ct.Solution('input-files/0d_example/mech.cti')
gas.TP = 682, 30.11E5
gas.set_equivalence_ratio(1.0, {'ch3och3': 1, 'c3h8': 1}, {'o2': 1, 'n2': 3.76})
gas.mole_fraction_dict()


# The experiment can be modeled as an ideal gas mixture
# =============================================================================
# the reactor can be modeled as constant volume for simplicity, 
# with no mass transfer to/from the environment. The reactor is an [`IdealGasReactor`]
# after creating the reactor, we add it to a [`ReactorNet`]
# The reactor network would also contain any flow controllers, if those were present.
# =============================================================================

# In[5]:


reac = ct.IdealGasReactor(gas)
netw = ct.ReactorNet([reac])


# The next step is to run the integrator. There are two options for this, 
#the [`advance`]
# and [`step`] methods of the `ReactorNet`. 

# =============================================================================
# The `advance` method is useful when there is a desired end time for the integration, 
# or you would like to have fixed step sizes in the output. 
# The only argument to the `advance` method is the desired end time, 
# and the solver will take as many internal, variable-sized, time steps as necessary to get to that end tim. 
# 
# By contrast, the `step` method takes one variable sized time step, 
# with the size determined by the integrator. 
# It is useful when you want as much detail (in terms of time history) as possible. 
# The easiest way to store the data is to append to a [`SolutionArray`]
# (https://cantera.org/documentation/docs-2.4/sphinx/html/cython/importing.html#cantera.SolutionArray).
# =============================================================================

# In[6]:


sol = ct.SolutionArray(gas, extra=["time", "volume"])

# End the simulation when the temperature exceeds 2500 K or the time reaches 0.1 s
while reac.T < 2500 and netw.time < 0.1:
    sol.append(time=netw.time, T=reac.T, P=reac.thermo.P, X=reac.thermo.X, volume=reac.volume)
    netw.step()


# In[7]:


plt.figure('Simulated Temperature')
plt.plot(sol.time, sol.T)
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]');


# In[8]:


plt.figure('Experimental Pressure vs. Simulated Pressure')
plt.plot(expt_time, expt_pres, label='Experimental')
plt.plot(sol.time + 0.032, sol.P/1.0E5, label='Simulation')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]');





# In[]
# ## Variable-Volume Simulation Setup

# The agreement between the experiment and the model is quite good, 
# =============================================================================
# although it is clear there is some experimental detail that is missing. 
# We can account for this detail (the compression stroke and post-compression heat-loss) 
# by changing the volume of the `IdealGasReactor` to model the processes occurring in the experiment. 
# 
# In Cantera, the volume of a `Reactor` is changed by specifying the velocity of a `Wall` attached to the reactor.
# =============================================================================
# 
# The basic RCM experiment is a piston-cylinder assembly. 
# The area of the piston is constant
# the velocity of the piston is equal to the rate of change of the volume of the cylinder.
# 
# $$\begin{aligned}
# V &= A h \\
# \frac{dV}{dt} &= A \frac{dh}{dt}
# \end{aligned}$$
# 
# Since we are not interested in any quantities such as the total heat release that would depend on the actual `Reactor` volume 
# it is sufficient to assume the piston has unit area. 
# Therefore, if we can find the rate of change of the volume of the cylinder in the experiment, 
# we can specify that as the velocity of the wall and create the appropriate volume change in the `Reactor`. 
# 
# From the experiment, we are able to compute a "volume trace," in this case, 
# by assuming isentropic compression and "expansion" 
#  using the measured pressure data to calculate the volume as a function of time. 
# =============================================================================
# This could also be done by directly measuring the position of the piston. 
# Rather than computing the derivative of the volume on every time step, 
# we would like to compute it once and store that result for speed. 
# To do so, we will define a Python `class` that computes $dV/dt$ when initialized 
# and returns the appropriate value when passed the time in the simulation. 
# This latter part is done with the special `__call__` method of the class.
# =============================================================================

# In[9]:


class VolumeProfile(object):
    """
    Set the velocity of the piston by using a user specified volume
    profile. The initialization and calling of this class are handled
    by the `cantera.Func1` interface of Cantera.
    The velocity is calculated by assuming a unit area and using the
    forward difference, calculated by `numpy.diff`. This function is
    only called once when the class is initialized at the beginning of
    a problem so it is efficient.
    Parameters
    ----------
    time: `numpy.ndarray`
        Array or list of time values
    volume: `numpy.ndarray`
        Array or list of volume values
    Attributes
    ----------
    time: `numpy.ndarray`
        Array of time values
    volume: `numpy.ndarray`
        Array of volume values
    velocity: `numpy.ndarray`
        Array of velocity values
    """

    def __init__(self, time, volume):
        # The time and volume are stored as lists in the keywords
        # dictionary. The volume is normalized by the first volume
        # element so that a unit area can be used to calculate the
        # velocity.
        self.time = np.array(time)
        self.volume = np.array(volume)/volume[0]

        # The velocity is calculated by the forward difference.
        # numpy.diff returns an array one element smaller than the
        # input array, so we append a zero to match the length of the
        # self.time array.
        self.velocity = np.diff(self.volume)/np.diff(self.time)
        self.velocity = np.append(self.velocity, 0)

    def __call__(self, t):
        """Return the velocity when called during a time step.
        Parameters
        ----------
        t : `float`
            Current simulation time.
        """

        if t <= self.time[-1] and t >= self.time[0]:
            # prev_time_point is the previous value in the time array
            # after the current simulation time
            prev_time_point = self.time[self.time <= t][-1]
            # index is the index of the time array where
            # prev_time_point occurs
            index = np.where(self.time == prev_time_point)[0][0]
            return self.velocity[index]
        else:
            return 0


# Now we need to define a `Solution` just like before.

# In[10]:


gas = ct.Solution('input-files/0d_example/mech.cti')
gas.TP = 323, 1.5137E5
gas.set_equivalence_ratio(1.0, {'ch3och3': 1, 'c3h8': 1}, {'o2': 1, 'n2': 3.76})


# The volume of the cylinder from the experiment is used to compute the velocity in the `VolumeProfile` class, 
# so we need to load the volume trace from the file on the disk.

# In[11]:


vol_data = np.loadtxt('input-files/0d_example/Tc_682K_P0_1.5137_T0_323K_volume.csv', delimiter=',')
inp_time = vol_data[:, 0]
inp_vol = vol_data[:, 1]


# We can calculate a few values of velocity using the `VolumeProfile` class, just to demonstrate the operation. 
# =============================================================================
# First, an instance of the class is initialized with the volume and time read from the input file. 
# Then we can call the instance, passing time points, and the velocity is returned. 
# If the time that is passed is greater than the maximum time present in the input file (0.4 s), the velocity returned is zero.
# =============================================================================

# In[12]:


vpro = VolumeProfile(inp_time, inp_vol)
print(vpro(0.01))
print(vpro(0.35))
print(vpro(0.5))


# To effect the volume change, instances of `Wall`s can be installed between two Cantera reactors. 
# =============================================================================
# In this case, we only care about reactions and changes in the reactor representing the RCM reaction chamber, 
# so we install a `Wall` between the `IdealGasReactor` and a `Reservoir`, which represents the environment. 
# Then we specify the `VolumeProfile` class as the means to calculate the velocity. 
# Other options would be a constant value, or a function that takes a single argument (the simulation time). 
# With the velocity defined and the `Wall` installed, the simulation can proceed as before.
# =============================================================================

# In[13]:


reac = ct.IdealGasReactor(gas)
env = ct.Reservoir(ct.Solution('air.xml'))
wall = ct.Wall(reac, env, A=1.0, velocity=vpro)
netw = ct.ReactorNet([reac])
netw.set_max_time_step(inp_time[1])

vol_sol = ct.SolutionArray(gas, extra=["time", "volume"])
# End the simulation when the temperature exceeds 2500 K or the time reaches 0.1 s
while reac.T < 2500 and netw.time < 0.1:
    vol_sol.append(time=netw.time, T=reac.T, P=reac.thermo.P, X=reac.thermo.X, volume=reac.volume)
    netw.step()


# We can compare a number of parameters between the simulation and experiment. 
# =============================================================================
# First, we can compare the volume that is read from the input file to the simulated volume of the reactor as a function of time. 
# Remember that we did not specify the volume of the reactor directly, 
# instead we specified it through the velocity. 
# The two volume traces agree exactly 
# (although the simulated volume is somewhat shorter than the input volume because of the conditions we placed on the integration).
# =============================================================================

# In[14]:


plt.figure('Volume Trace Comparison')
plt.plot(inp_time, inp_vol, label='Input Volume')
plt.plot(vol_sol.time, vol_sol.volume, label='Simulated Volume')
plt.ylabel('Volume [m**3]')
plt.xlabel('Time, [s]')
plt.legend();


# In[15]:


plt.figure('Experimental Pressure vs. Simulated Pressure 2')
plt.plot(expt_time, expt_pres, label='Experimental')
plt.plot(sol.time + 0.032, sol.P/1.0E5, label='Constant Volume Simulation')
plt.plot(vol_sol.time, vol_sol.P/1.0E5, label='Variable Volume Simulation')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]');


# Then, comparing all three pressure traces, 
# =============================================================================
# we can see that the green variable volume simulation very closely 
# approximates the behavior of the experimental pressure trace, 
# including the compression stroke and post-compression behavior. 
# User defined functions and classes, such as the one used to compute the velocity of the wall in this example, 
# are very powerful in Cantera.
# =============================================================================
