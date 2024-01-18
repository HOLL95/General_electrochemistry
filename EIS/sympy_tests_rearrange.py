import sympy
z=["R1", "Cdl",  "omega", "Cf", "Rs"]
var={key:sympy.var(key) for key in z}
var["i"]=sympy.var("i", complex=True, real=False)
expression=(1/((var["i"]*var["omega"]*var["Cdl"])+((var["i"]*var["omega"]*var["Cf"])/(1+var["i"]*var["omega"]*var["Cf"]*var["R1"]))))
#sympy.pprint(expression)
sympy.pprint(sympy.simplify(expression))
print("~"*100)
horrible_expression=(var["Cdl"]*var["Cf"]*var["R1"]*(var["omega"]**2)-var["i"]*var["omega"]*var["Cdl"]-var["i"]*var["omega"]*var["Cf"])#/(var["Cdl"]*var["Cf"]*var["R1"]*(var["omega"]**2)-var["i"]*var["omega"]*var["Cdl"]-var["i"]*var["omega"]*var["Cf"])
expression_1=expression*horrible_expression#*1/horrible_expression
sympy.pprint(expression_1)
sympy.pprint(expression_1/horrible_expression)