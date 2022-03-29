import sympy
z=["R1", "Q1", "alpha1", "i", "omega", "s"]
variables={key:sympy.var(key) for key in z}
expression=variables["i"]*variables["R1"]/(1+variables["R1"]+variables["Q1"]*(variables["s"]**variables["alpha1"]))
print(sympy.apart(expression))
