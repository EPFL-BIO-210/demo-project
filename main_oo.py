# Code adapted from https://github.com/scipy/scipy-cookbook/blob/master/ipython/LotkaVolterraTutorial.ipynb

import numpy as np
from Visualization import evolution

class DataSaver:
    """ Class to store data from the model """
    def __init__(self, instance=None):
        # self.a = LVM.a
        self.data = {"state_X": [], "state_T": []}
        if instance is not None:  # to store parameters with the result
            self.a = instance.a
            self.b = instance.b
            self.c = instance.c
            self.d = instance.d
            self.dt = instance.dt

    def reset(self):
        """ Reset the stored data """
        self.data = {"state_X": [], "state_T": []}

    def store_iter(self, X:np.ndarray=None, T:float=None):
        """ Store one iteration of the model """
        if X is not None:
            self.data["state_X"].append(X.copy())
        if T is not None:
            self.data["state_T"].append(T)

    def get_data(self):
        """ Return the stored data """
        return self.data

    def LyapunovFunction():
        raise NotImplementedError()
    
class LVM:
    """ Simple LotkaVolterraClass with Euler Integration 
    input:
        a: natural growth rate of preys
        b: natural dying rate of preys
        c: natural dying rate of predators
        d: factor describing growth of predators based on caught preys
        dt: time step
    """
    def __init__(self, a:float=1.0, b:float=0.1, c:float=1.5, d:float=0.75, dt:float=0.1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt

    def update(self, X:np.ndarray):
        """ Forward Euler method update """
        return X + self.dt * np.array(
            [self.a * X[0] - self.b * X[0] * X[1],
                -self.c * X[1] + self.d * self.b * X[0] * X[1]])

    def dynamics(self, X0:np.ndarray, num_iter:int, saver:DataSaver):    # saver defined later!
        """ Run each iteration of the model and store the results 
        input:
            X0: initial conditions
            num_iter: number of iterations
            saver: instance of DataSaver
        """
        X = X0  # initalize
        saver.store_iter(X, 0)
        for i in range(num_iter):
            X = self.update(X)
            saver.store_iter(X, self.dt * i)


if __name__ == "__main__":
    # Definition of parameters
    gr_rate_prey = 1.0  # natural growth rate of rabbits (prey)
    dy_rate_prey = 0.1  # natural dying rate of rabbits
    dy_rate_predators = 1.5  # natural dying rate of foxes
    gr_rate_predators = 0.75  # factor describing growth of foxes based on caught rabbits

    numiter = 10000
    dt = 15 * 1.0 / numiter

    X0 = np.array([10, 5])        # initial conditions

    lvm = LVM(gr_rate_prey, dy_rate_prey, dy_rate_predators, gr_rate_predators, dt)     # Creating model instance
    saver = DataSaver(lvm)        # Creating saving instance
    # Note: passing instance to store parameters!

    lvm.dynamics(X0, numiter, saver)    # Running the dynamics and storing in saver instance

    T, Xeuler = saver.get_data()["state_T"], np.array(saver.get_data()["state_X"])
    rabbits, foxes = Xeuler.T

    evolution(T, Xeuler)
