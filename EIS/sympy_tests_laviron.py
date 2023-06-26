import sympy as sym
import numpy as np
sym.init_printing()
z=[ "area", "alpha", "k0", "gamma_red", "gamma_ox", "norm_E", "R", "T", "F"]
var={key:sym.var(key) for key in z}
var["Ca"]=sym.var("Ca")
var["Ra"]=sym.var("Ra")

eqn1=sym.Eq((var["R"]*var["T"]/(var["F"]**2*var["area"]*var["k0"]))*(1/((var["gamma_ox"]*var["alpha"]*sym.exp((var["F"]/(var["R"]*var["T"]))*(-var["alpha"])*var["norm_E"]))+(var["gamma_red"]*(1-var["alpha"])*sym.exp((var["F"]/(var["R"]*var["T"]))*(1-var["alpha"])*var["norm_E"])))), var["Ra"])
eqn2=sym.Eq(1/(var["k0"]*var["Ra"]*(sym.exp((var["F"]/(var["R"]*var["T"]))*(-var["alpha"])*var["norm_E"])+sym.exp((var["F"]/(var["R"]*var["T"]))*(1-var["alpha"])*var["norm_E"]))),var["Ca"])
sym.pprint(eqn1)
sym.pprint(eqn2)

#result= sym.solve([eqn1, eqn2],(var["k0"], var["alpha"], var["gamma_ox"]))
#print(result)
F=96485.3321
R=8.3145
T=298
RT=R*T
FRT=F/(R*T)
k0=100
e0=0.001
alpha=0.55
gamma=1e-10
area=0.07
dc_pot=0
ratio=np.exp(FRT*(e0-dc_pot))
red=gamma/(ratio+1)
ox=gamma-red
Ra_coeff=(R*T)/((F**2)*area*k0)
nu_1_alpha=np.exp((1-alpha)*FRT*(dc_pot-e0))
nu_alpha=np.exp((-alpha)*FRT*(dc_pot-e0))
Ra=Ra_coeff*((alpha*ox*nu_alpha)+((1-alpha)*red*nu_1_alpha))**-1
sigma=k0*Ra*(nu_alpha+nu_1_alpha)
Ca=1/sigma
norm_E=dc_pot-e0
gamma_ox=ox
gamma_red=red
exp=np.exp
vals=dict(zip(z, [area, alpha, k0, red, ox, dc_pot-e0, R, T, F]))
sRa=eqn1.evalf(subs=vals)

sCa=eqn2.evalf(subs=vals)
#Ra=float(sRa.lhs)


#Ca=float(sCa.evalf(subs={"Ra":Ra}).lhs)
#print(Rf, Cf)
print(Ra, Ca)
z=[(exp(norm_E*(Ca*RT*exp(F*norm_E/RT) + Ca*RT - F**2*area*gamma_red*exp(F*norm_E/RT))/(F*RT*area*(gamma_ox - gamma_red*exp(F*norm_E/RT))))/(Ca*Ra*(exp(F*norm_E/RT) + 1)), (Ca*RT*exp(F*norm_E/RT) + Ca*RT - F**2*area*gamma_red*exp(F*norm_E/RT))/(F**2*area*(gamma_ox - gamma_red*exp(F*norm_E/RT))))]

print(z)