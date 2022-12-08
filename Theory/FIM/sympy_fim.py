import sympy as sym
info_symbols={x:sym.symbols(x) for x in ["E_0", "k_0", "Ru", "Cdl", "CdlE1", "CdlE2", "CdlE3", "gamma", "alpha"]}
other_symbols={x:sym.symbols(x) for x in ["E_start", "dI", "deltaE"]}
variables={x:sym.symbols(x) for x in ["I", "t" ]}
num_sines=4
e_keys=["E_{0}".format(i) for i in range(1,num_sines+1) ]+["dE_{0}".format(i) for i in range(1,num_sines) ]
E_vars={x:sym.symbols(x) for x in e_keys}

e_equations={x:sym.Eq(x, variables["fE"], other_symbols["E_start"]+other_symbols["deltaE"]*sym.sin(other_symbols["omega"]*variables["t"]+info_symbols["eta"]))}


E=sym.Eq(variables["fE"], other_symbols["E_start"]+other_symbols["deltaE"]*sym.sin(other_symbols["omega"]*variables["t"]+info_symbols["eta"]))

dE=sym.Eq(other_symbols["deltaE"]*sym.cos(other_symbols["omega"]*variables["t"]+info_symbols["eta"]))
