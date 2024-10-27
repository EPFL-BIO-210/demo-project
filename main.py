# Code adapted from https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html
import numpy as np
from scipy import integrate
from functions import dx_dt


# Definition of parameters
a = 1.0  # natural growth rate of rabbits (prey)
b = 0.1  # natural dying rate of rabbits
c = 1.5  # natural dying rate of foxes
d = 0.75  # factor describing growth of foxes based on caught rabbits

t = np.linspace(0, 15, 1000)  # time
X0 = np.array([10, 5])  # initial conditions: 10 rabbits and 5 foxes
X, infodict = integrate.odeint(lambda x, _: dx_dt(x, a, b, c, d), X0, t, full_output=True)

print(X)
