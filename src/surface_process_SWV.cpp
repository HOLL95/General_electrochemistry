#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>
namespace py = pybind11;
using namespace std;
template <typename V>
V get(py::dict m, const std::string &key, const V &defval) {
    return m[key.c_str()].cast<V>();
}


double potential(int j, double SF, double dE, double Esw, double E_start){
  double first_term=ceil((j/(SF/2))*0.5)*dE;
  int second_term;
  if (ceil(j/(SF/2))/2==ceil((j/(SF/2))*0.5)){
    second_term=1;
  }else{
    second_term=-1;
  }
  return E_start+Esw-(((first_term+second_term*Esw)+Esw)-dE);
}
double fourier_Et(double pi, double scan_rate, double order, double deltaE, double E_start, double t){
  double E=E_start;
  double coeff=(4*deltaE)/pi;
  double denom;
  for (int i = 1; i <= order; i+=2) {
    denom=1/(i*1.0);
    E+=((denom*std::sin(t*i)));
  }
  return coeff*E+(scan_rate*t);
}
double fourier_dEdt(double pi, double scan_rate, int order, double deltaE, double t){
  double dEdt=0;
  double coeff=(4*deltaE)/pi;
  for (int i = 1; i <= order; i+=2) {
    dEdt+=(std::cos(t*i));
  }
  return coeff*dEdt;
}

py::object SWV_current(py::dict params, std::vector<double> t, std::string method, double debug=-1, double bounds_val=10,const bool Marcus_flag=false, const bool interaction_flag=false) {
    double E;
    double numerator, denominator;
    double Itot_sum;
    const double F=96485.3328959;
    const double R=8.314459848;
    const double T = get(params,std::string("T"),298);
    const double n = get(params,std::string("n"),1);
    const double pi=3.14159265358979323846;
    const double FRT= (n*F)/(R*T);
    const double k0 = get(params,std::string("k_0"),35.0);
    const double alpha = get(params,std::string("alpha"),0.5);
    const double gamma = get(params,std::string("gamma"),1.0);
    const double E0 = get(params,std::string("E_0"),0.25);
    const double Ru = get(params,std::string("Ru"),0.001);
    const double Es = get(params,std::string("E_start"),-10.0);
    const double d_E = get(params,std::string("scan_increment"),0.1);
    const double delta_E = get(params,std::string("deltaE"),0.1);
    const int SF= get(params,std::string("sampling_factor"),0.1);
    const double Esw= get(params,std::string("SW_amplitude"),0.1);
    const double E_start=Es-E0;
    int end =(delta_E/d_E)*SF;
    std::vector<double> Itot(end, 0);
    E=FRT*(potential(0, SF, d_E, Esw, E_start));
    Itot[0]=k0*exp(-alpha*E)/(1+(1+exp(E))*((k0*exp(-alpha*E))/SF));
    Itot_sum=Itot[0];
    for (int j = 1; j <= end; j++) {
      E=FRT*(potential(j, SF, d_E, Esw, E_start));
      numerator=k0*exp(-alpha*E)*(1-((1+exp(E))/SF)*Itot_sum);
      denominator=1+((k0*exp(-alpha*E)/SF)*(1+exp(E)));
      Itot[j-1]=numerator/denominator;
      Itot_sum+=Itot[j-1];
    }
    for (int j=0; j<end;, j++){
      Itot[j]=Itot[j]*gamma;

    }
    return py::cast(Itot);
  }




PYBIND11_MODULE(SWV_surface, m) {
	m.def("SWV_current", &SWV_current, "solve for I_tot using recurrence method for SWV");
  m.def("potential", &potential, "SWV potential function");
  m.def("fourier_Et", &fourier_Et, "Truncated fourier approximation for SWV potential func");
  m.def("fourier_dEdt", &fourier_dEdt, "Truncated fourier approximation for derivative of SWV potential func");
}
