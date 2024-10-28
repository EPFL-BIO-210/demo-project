"""
Module containing simulation code of the Lotka Volterra model.

Adapted from:
https://github.com/scipy/scipy-cookbook/blob/master/ipython/LotkaVolterraTutorial.ipynb

"""

import numpy as np


def dX_dt(X, a=1.0, b=0.1, c=1.5, d=0.75):
    """
    Computes the growth rate of fox and rabbit populations based on system state (X) and parameters (a,b,c,d)

    Parameters
    ----------
    X : array or tuple
        [prey_count, predator_count]
    a : float, optional
        natural growth rate of the prey (rabbit)
    b : float, optional
        natural dying rate of the prey
    c : float, optional
        natural growth rate of the predator (fox)
    d : float, optional
        natural dying rate of the predator

    Returns
    -------
    numpy array
        [change of prey_count, change of predator_count]
        
    Examples
    -------
    >>> dX_dt(np.ones(2),1,0.1,1.5,.75)
    array([ 0.9  , -1.425])
    >>> dX_dt(np.zeros(2),1,0.1,1.5,.75)        # zero is a fixpoint
    array([0., 0.])
    """
    if np.size(X) > 2:
        raise ValueError("X has only two dimensions!")

    return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])


