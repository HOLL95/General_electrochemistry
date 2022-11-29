import sympy
sympy.init_printing() 
syms=["alpha", "omega", "tau"]
sym_dict={key:sympy.Symbol(key) for key in syms}
expression=(1j*sym_dict["omega"]*sym_dict["tau"])**(1-sym_dict["alpha"])/(1+((1j*sym_dict["omega"]*sym_dict["tau"])**(1-sym_dict["alpha"])))
Q=1e-3
Rf=10
Re=20
alpha=0.5
omega=15
Expression_1=1/(Re+(1/(1/Rf)+Q*(1j*omega)**alpha))
Expression_2_num=1+Rf*Q*(1j*omega)**alpha
Expression_2_denom=Rf+Re+Re*Rf*Q*(1j*omega)**alpha
Expression_2=Expression_2_num/Expression_2_denom
Expression_3=1-(Rf/(Re+Rf))*(1/(1+(Re*Rf/(Re+Rf))*(Q*(1j*omega)**alpha)))
Expression_3=Expression_3*1/Re
print(Expression_1)
print(Expression_2)
print(Expression_3)