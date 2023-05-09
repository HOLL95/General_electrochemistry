import arviz as az

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

print(f"Running on PyMC v{pm.__version__}")
def my_model(theta, x):
    m, c = theta
    return m * x + c


def my_loglike(theta, x, data, sigma):
    model = my_model(theta, x)
    model_2 = my_model(theta, x)
    lh_1=-(0.5 / sigma**2) * np.sum((data - model) ** 2)
    lh_2=-(0.5 / sigma**2) * np.sum((data - model) ** 2)
    return lh_1+lh_2
# define a pytensor Op for our likelihood function
class LogLike(pt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):

        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)
        outputs[0][0] = np.array(logl)  # output the log-likelihood

# set up our data
N = 10  # number of data points
sigma = 1.0  # standard deviation of noise
x = np.linspace(0.0, 9.0, N)

mtrue = 0.4  # true gradient
ctrue = 3.0  # true y-intercept

truemodel = my_model([mtrue, ctrue], x)

# make data
rng = np.random.default_rng(716743)
data = sigma * rng.normal(size=N) + truemodel

# create our Op
logl = LogLike(my_loglike, data, x, sigma)

# use PyMC to sampler from log-likelihood
for j in range(0, 1):
    with pm.Model():
        # uniform priors on m and c
        #m = pm.Uniform("m", lower=-10.0, upper=10.0)
        #c = pm.Uniform("c", lower=-10.0, upper=10.0)
        #print(vars(m))
        v=[pm.Uniform(x, lower=-10, upper=10) for x in ["m"]]
        q=[pm.Uniform(x, lower=-10, upper=10) for x in ["c"]]
        # convert m and c to a tensor vector
        theta = pt.as_tensor_variable(v+q)
        save_str= "/home/henney/Documents/Oxford/General_electrochemistry/heuristics/file_{0}.nc".format(j)

        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))

        # Use custom number of draws to replace the HMC based defaults
        idata_mh = pm.sample(3000, tune=1000)
        idata_mh.to_netcdf(save_str)

# plot the traces
az.plot_trace(idata_mh, lines=[("m", {}, mtrue), ("c", {}, ctrue)])
