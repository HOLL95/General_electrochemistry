import sympy as sym
param_vars={x:sym.symbols(x) for x in ["E_0", "k_0", "Ru", "Cdl", "CdlE1", "CdlE2", "CdlE3", "gamma", "alpha"]}
deriv_vars={x:sym.symbols(x) for x in [ "dI", "dE", "dtheta"]}
state_vars={x:sym.symbols(x) for x in ["I", "theta", "E"]}
t=sym.symbols("t")
num_sines=4
#dimensional_params=[sym.symbols(x) for x in ]
sinusoid_params=[]
labels=["freq", "amp", "phase"]
for i in range(0, num_sines):          
            sinusoid_params+=[x+"_{0}".format(i+1) for x  in labels]
sinusoid_vars={x:sym.symbols(x) for x in sinusoid_params}

E_vars={x:sym.symbols(x) for x in ["E_{0}".format(i) for i in range(1,num_sines+1) ]}
dE_vars={x:sym.symbols(x) for x in ["dE_{0}".format(i) for i in range(1,num_sines+1)]}

full_expr=param_vars["Cdl"]* \
                (1+param_vars["CdlE1"]*(state_vars["E"]-(param_vars["Ru"]*state_vars["I"]))+\
                param_vars["CdlE2"]*(state_vars["E"]-(param_vars["Ru"]*state_vars["I"]))**2+\
                param_vars["CdlE3"]*(state_vars["E"]-(param_vars["Ru"]*state_vars["I"]))**3)*\
                (deriv_vars["dE"]-param_vars["Ru"]*deriv_vars["dI"])+\
                param_vars["gamma"]*\
                (param_vars["k_0"]*(1-state_vars["theta"])*sym.exp((1-param_vars["alpha"])*(state_vars["E"]-param_vars["E_0"]-(state_vars["I"]*param_vars["Ru"])))-\
                param_vars["k_0"]*(state_vars["theta"])*sym.exp((-param_vars["alpha"])*(state_vars["E"]-param_vars["E_0"]-(state_vars["I"]*param_vars["Ru"]))))
print(full_expr)
current_eq=sym.Eq(state_vars["I"], full_expr)
print(sym.solve(current_eq, deriv_vars["dI"]))
print("dI equation")


#current_eq.subs(deriv_vars["dtheta"],  
#                param_vars["k_0"]*(1-state_vars["theta"])*sym.exp((1-param_vars["alpha"])*(state_vars["E"]-param_vars["E_0"]-state_vars["I"]*param_vars["Ru"]))-
#                param_vars["k_0"]*(state_vars["theta"])*sym.exp((param_vars["alpha"])*(state_vars["E"]-param_vars["E_0"]-state_vars["I"]*param_vars["Ru"])))
#er=sym.symbols("Er")
#current_eq.subs("Cdl", param_vars["Cdl"]*(1+param_vars["CdlE1"]*(state_vars["E"]-(param_vars["Ru"]*state_vars["I"]))+param_vars["CdlE2"]*(state_vars["E"]-(param_vars["Ru"]*state_vars["I"]))**2+param_vars["CdlE1"]*(state_vars["E"]-(param_vars["Ru"]*state_vars["I"]))**3))


#current_eq.subs("E",
#              sym.Add(*[sinusoid_vars["amp_{0}".format(x)]*sym.sin(sinusoid_vars["freq_{0}".format(x)]*t+sinusoid_vars["phase_{0}".format(x)]) for x in range(1, num_sines+1)])
#                )
#current_eq.subs("dE",
#               sym.Add(*[sinusoid_vars["amp_{0}".format(x)]*sinusoid_vars["freq_{0}".format(x)]*sym.cos(sinusoid_vars["freq_{0}".format(x)]*t+sinusoid_vars["phase_{0}".format(x)]) for x in range(1, num_sines+1)])
#                )
#sym.pprint(current_eq)
#sym.pprint(sym.solve(current_eq, deriv_vars["dI"]))
counter=0
for key in param_vars.keys():
    print("#", key)
    print("sensitivities[:,{0}]=".format(counter), sym.diff(full_expr, param_vars[key]))
    counter+=1
print()