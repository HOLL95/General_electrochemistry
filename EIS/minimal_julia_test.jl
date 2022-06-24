
using DifferentialEquations

function fiip(du,u,p,t)
  println(p)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
  z=rand((0, 10))
  println(z)
  p[1]=z
  println(p)
  println("-")
end
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5())
