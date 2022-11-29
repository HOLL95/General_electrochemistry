import numpy as np
import matplotlib.pyplot as plt
import sys
harm_range=list(range(4, 6))
from scipy import interpolate
from SALib.sample import saltelli
from SALib.analyze import sobol
def parabola(x, a, b):
    """Return y = a + b*x**2."""
    return a + b*x**2



problem = {
    'num_vars': 2,
    'names': ['a', 'b'],
    'bounds': [[0, 1]]*2
}


# sample
param_values = saltelli.sample(problem, 2**6)

# evaluate
x = np.linspace(-1, 1, 100)
y = np.array([parabola(x, *params) for params in param_values])
print(y.T)
# analyse
sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]

