#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/math/tools/minima.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/tuple.hpp>
#include <math.h>
#include <iostream>
#include <exception>
#include <gsl/gsl_integration.h>
namespace py = pybind11;
using namespace std;
template <typename V>
V get(py::dict m, const std::string &key, const V &defval) {
    return m[key.c_str()].cast<V>();
}
struct Integration_parameters {
    double Upper_lambda;
    double Normalised_E;
};
double Marcus_integral_reduction(double x, void * params){
    
    struct Integration_parameters *p = (struct Integration_parameters *) params;
    double Upper_lambda = p->Upper_lambda;
    double Normalised_E=p->Normalised_E;
    double numerator=exp(-(Upper_lambda/4)*pow(1-((Normalised_E+x)/Upper_lambda),2));  
    double f = numerator/(1+exp(x));
return f;
}
double Marcus_integral_oxidation(double x,void * params){
    struct Integration_parameters  *p = (struct Integration_parameters  *) params;
    double Upper_lambda = p->Upper_lambda;
    double Normalised_E=p->Normalised_E;
    double numerator=exp(-(Upper_lambda/4)*pow(1+((Normalised_E+x)/Upper_lambda),2));  
    double f = numerator/(1+exp(-x));
return f;
}
double Marcus_kinetics(double Normalised_E,double Upper_lambda, int flag){
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    gsl_function F;
    struct Integration_parameters params;
    params.Upper_lambda=Upper_lambda;
    params.Normalised_E=Normalised_E;
    if (flag==0){
        F.function = &Marcus_integral_oxidation;
    }
    else if (flag==1){
        F.function = &Marcus_integral_reduction;
    }
    F.params=&params;
    //F.params = &params; // Pass a pointer to the struct

    double result, error;
    gsl_integration_qag(&F, -50, 50, 0, 1e-7, 1000,5, w, &result, &error);

    //onst gsl_function *f, double a, double b, double epsabs, double epsrel, size_t limit, int key, gsl_integration_workspace *workspace, double *result, double *abse
    //const gsl_function *f, double a, double b, double epsabs, double epsrel, size_t limit, gsl_integration_workspace *workspace, double *result, double *abserr)
    //TODO:1)add integral0 and lambda to C++ 2) compile 3)test 4) check single sinewave int calculations
    gsl_integration_workspace_free(w);

    return result;
}
struct e_surface_fun {
    double E,dE;
    double cap_E;
    double Cdl,CdlE,CdlE2,CdlE3;
    double E0;
    double Ru;
    double k0;
    double alpha;
    double In0,u1n0;
    double dt;
    double gamma;
    double exp11,exp12;
    double dexp11,dexp12;
    double u1n1;
    double du1n1;
    double Cdlp;
    double Upper_lambda;
    double integral_0_oxidation;
    double integral_0_reduction;
    double aoo, arr, aor, k0_ap, E0_ap, S, G; 
    bool Marcus_flag;
    bool interaction_flag;

    e_surface_fun (
                    const double E,
                    const double dE,
                    const double cap_E,
                    const double Cdl,
                    const double CdlE,
                    const double CdlE2,
                    const double CdlE3,
                    const double E0,
                    const double Ru,
                    const double k0,
                    const double alpha,
                    const double In0,
                    const double u1n0,
                    const double dt,
                    const double gamma,
                    const double Upper_lambda,
                    const double integral_0_oxidation,
                    const double integral_0_reduction,
                    const bool Marcus_flag,
                    const double aoo,
                    const double arr,
                    const double aor, 
                    const double k0_ap, 
                    const double E0_ap, 
                    const double S,
                    const double G,
                    const bool interaction_flag
                    ) :
        E(E),dE(dE),cap_E(cap_E),Cdl(Cdl),
        CdlE(CdlE),CdlE2(CdlE2),CdlE3(CdlE3),E0(E0),Ru(Ru),
        k0(k0),alpha(alpha),In0(In0),u1n0(u1n0),dt(dt),gamma(gamma), 
        Upper_lambda(Upper_lambda),integral_0_oxidation(integral_0_oxidation),
        integral_0_reduction(integral_0_reduction),Marcus_flag(Marcus_flag),
        aoo(aoo), arr(arr), k0_ap(k0_ap), E0_ap(E0_ap), S(S), G(G), interaction_flag(interaction_flag)
         { }
  //boost::math::tuple<double,double> operator()(const double In1) {
        //update_temporaries(In1);
        //return boost::math::make_tuple(residual(In1),residual_gradient(In1));
    //}
    double operator()(double const In1){
      update_temporaries(In1);
      return abs(residual(In1));
    }

    double residual(const double In1) const {
        return Cdlp*(dt*dE-Ru*(In1-In0)) - dt*In1 + gamma*(u1n1-u1n0);
        //return Cdlp*(dt*dE) - dt*In1 + (u1n1-u1n0) + Ru*E*dt;
    }

    double residual_gradient(const double In1) const {
        return -Cdlp*Ru - dt + gamma*du1n1;
        //return -Cdlp*Ru - dt + du1n1;
    }


    void update_temporaries(const double In1) {
        const double Ereduced = E - Ru*In1;
        const double cER=cap_E-Ru*In1;
        //const double Ereduced = E;
        const double Ereduced2 = pow(cER,2);
        const double Ereduced3 = cER*Ereduced2;
        const double expval1 = Ereduced - E0;
        //exp11 = std::exp((1.0-alpha)*expval1);
        //exp12 = std::exp(-alpha*expval1);
        
        if (Marcus_flag==true){

            exp11 = Marcus_kinetics(expval1,Upper_lambda, 1)/integral_0_reduction;//std::exp((1.0-alpha)*expval1);
            exp12 = Marcus_kinetics(expval1,Upper_lambda, 0)/integral_0_oxidation;//std::exp(-alpha*expval1);
        }
        else if (Marcus_flag==false){
            exp11 = std::exp((1.0-alpha)*expval1);
            exp12 = std::exp(-alpha*expval1);
        }
        if (interaction_flag==true){

        }
        else if (interaction_flag==false){
        const double u1n1_top = dt*k0*exp11 + u1n0;
        const double denom = (dt*k0*exp11 +dt*k0*exp12 + 1);
        const double tmp = 1.0/denom;
        const double tmp2 = pow(tmp,2);
        u1n1 = u1n1_top*tmp;
        }
        
        Cdlp = Cdl*(1.0 + CdlE*cER + CdlE2*Ereduced2 + CdlE3*Ereduced3);
    
    }
};
double et(double E_start, double omega, double phase, double delta_E,double t){
	double E_t=(E_start+delta_E)+delta_E*(std::sin((omega*t)+phase));

	return E_t;
}

double dEdt(double omega, double phase, double delta_E, double t){
  double dedt=(delta_E*omega*std::cos(omega*t+phase));
	return dedt;
}
double c_et(double E_start, double E_reverse,double tr, double omega, double phase, double v, double delta_E, double t){
	double E_dc;
	double E_t;
	if (t<tr){
		E_dc=E_start+(v*t);
	}else {
		E_dc=E_reverse-(v*(t-tr));
	}

	 E_t=E_dc+(delta_E*(sin((omega*t)+phase)));

	return E_t;
}

double c_dEdt( double tr, double omega, double phase, double v, double delta_E, double t){ //
	double E_dc;
	if (t < tr){
		 E_dc=v;
	}else {
		 E_dc=-v;
	}
  double dedt= E_dc+(delta_E*omega*cos(omega*t+phase));

	return dedt;
}
double dcv_et(double E_start, double E_reverse,double tr, double v,  double t){
	double E_dc;
	if (t<tr){
		E_dc=E_start+(v*t);
	}else {
		E_dc=E_reverse-(v*(t-tr));
	}
	return E_dc;
}

double dcv_dEdt( double tr, double v, double t){ //
	double E_dc;
	if (t < tr){
		 E_dc=v;
	}else {
		 E_dc=-v;
	}
  double dedt= E_dc;

	return dedt;
}
double sum_of_sinusoids_E(const std::vector<double> &amplitudes,const std::vector<double> &frequencies,const std::vector<double> &phases,const int num_frequencies, double t){
  double E=0;
  for (int i=0; i<num_frequencies;i++){
    E+=amplitudes[i]*sin((frequencies[i]*t)+phases[i]);
  }
  return E;
}
double sum_of_sinusoids_dE(const std::vector<double> &amplitudes,const std::vector<double> &frequencies,const std::vector<double> &phases,const int num_frequencies, double t){
  double dE=0;
  for (int i=0; i<num_frequencies;i++){
    dE+=amplitudes[i]*frequencies[i]*cos((frequencies[i]*t)+phases[i]);
  }
  return dE;
}
std::vector<vector<double>> NR_function_surface(e_surface_fun &bc, double I_0, double I_minus, double I_bounds){
  cout<<"called"<<"\n";
  double interval=0.01;
  int width=(1/interval)*10;
  double start=I_0-(width*interval);
  std::vector<vector<double>> diagnostic;
  diagnostic.resize(4, std::vector<double> ((width*2)+1));
  for (int i=0; i<((width*2)+1);i++){
    diagnostic[0][i]=start+interval*i;
    if(i==(width+1)){
      cout<<I_0<<"\n";
      diagnostic[0][i]=I_0;
      bc.update_temporaries(I_0);
      diagnostic[1][i]=bc.residual(I_0);
      diagnostic[2][i]=bc.residual_gradient(I_0);
    }else{
      bc.update_temporaries(start+interval*i);
      diagnostic[1][i]=bc.residual(start+interval*i);
      diagnostic[2][i]=bc.residual_gradient(start+interval*i);
    }
  }
  diagnostic[3][0]=I_minus;
  diagnostic[3][1]=I_0;
  diagnostic[3][2]=I_bounds;
  return diagnostic;
}


py::object brent_current_solver(py::dict params, std::vector<double> t, std::string method, double debug=-1, double bounds_val=10, const bool Marcus_flag=false, const bool interaction_flag==false) {
    const double v=1;
    const int digits_accuracy = std::numeric_limits<double>::digits;
    const double max_iterations = 100;
    double E, dE, cap_E;
    int input=-1; // 0 for ramped, 1 for sinusoidal, 2 for DCV
    const double pi = boost::math::constants::pi<double>();
    //cout<<method<<"\n";
    if ((method.compare("ramped"))==0){
      input=0;
    }
    else if ((method.compare("sinusoidal"))==0){
      input =1;
    }
    else if ((method.compare("dcv"))==0){
      input =2;
    }
    else if ((method.compare("sum_of_sinusoids"))==0){
      input =3;
    }
    else{
      throw std::runtime_error("Input voltage method not defined");
    }

    const int num_frequencies = get(params, std::string("num_frequencies"), 5);
    const std::vector<double> freq_vector= get(params, std::string("freq_array"), std::vector<double> (5, 0));
    const std::vector<double> amp_vector= get(params, std::string("amp_array"), std::vector<double> (5, 0));
    const std::vector<double> phase_vector= get(params, std::string("phase_array"), std::vector<double> (5, 0));
    const double omega = get(params,std::string("nd_omega"),2*pi);
    const double phase = get(params,std::string("phase"),0.0);
    const double cap_phase = get(params,std::string("cap_phase"),0.0);
    const double delta_E = get(params,std::string("d_E"),0.1);
    const double E_start = get(params,std::string("E_start"),-10.0);
    const double E_reverse = get(params,std::string("E_reverse"),10.0);
    double tr=E_reverse-E_start;
  
    std::vector<double> Itot(t.size(), 0);
    const double k0 = get(params,std::string("k_0"),35.0);
    const double alpha = get(params,std::string("alpha"),0.5);
    const double gamma = get(params,std::string("gamma"),1.0);
    const double E0 = get(params,std::string("E_0"),0.25);
    const double Ru = get(params,std::string("Ru"),0.001);
    double Cdl = get(params,std::string("Cdl"),0.0037);
    double CdlE = get(params,std::string("CdlE1"),0.0);
    double CdlE2 = get(params,std::string("CdlE2"),0.0);
    double CdlE3 = get(params,std::string("CdlE3"),0.0);
    double integral_0_reduction=0;
    double integral_0_oxidation=0;
    double Upper_lambda=0;
    double aoo, aor, arr;
    if (Marcus_flag==true){
      Upper_lambda=get(params,std::string("Upper_lambda"),0.0);
      integral_0_reduction=Marcus_kinetics(0, Upper_lambda, 1);
      integral_0_oxidation=Marcus_kinetics(0, Upper_lambda, 0);
    }
    if (interaction_flag==true){
      aoo=get(params,std::string("aoo"),0.0);
      aor=get(params,std::string("aor"),0.0);
      arr=get(params,std::string("arr"),0.0);
      const double S=arr-aoo;
      const double G=aoo+arr-(2*aor);
      const double k0_ap=k0*exp(-2*gamma*aoo)
      const double E0_ap=get(params,std::string("E0_ap"),0.0);


    }
    
    
    const double sf= get(params,std::string("sampling_freq"),0.1);
    const double dt=  min(t[1]-t[0], sf);
    double Itot0,Itot1;
    double u1n0, theta_0;
   
    double t1 = 0.0;
    u1n0 = 0.0;
    double Cdlp = Cdl*(1.0 + CdlE*E + CdlE2*pow(E,2)+ CdlE3*pow(E,2));
    Itot0 =Cdlp*dE;

    
    
   
    
    if (input==0){
      E = et(E_start, omega, phase,delta_E ,t1+dt);
      dE = dEdt(omega, cap_phase,delta_E , t1+0.5*dt);
    }
    else if (input ==1){
      E = c_et( E_start, E_reverse, tr, omega,  phase,  v,  delta_E, t1);
      dE = c_dEdt(tr,  omega,  cap_phase,  v,  delta_E, t1+0.5*dt);
    }
    else if (input==2){
      E=dcv_et(E_start, E_reverse,tr,v, t1);
      dE=dcv_dEdt(tr,v, t1+0.5*dt);

    }
    else if (input==3){
      E=sum_of_sinusoids_E(amp_vector, freq_vector, phase_vector, num_frequencies, t1);
      dE=sum_of_sinusoids_dE(amp_vector, freq_vector, phase_vector, num_frequencies, t1);;

    }
    //cout<<E<<" "<<dE<<" "<<" "<<Cdl<<" "<<CdlE<<" "<<CdlE2<<" "<<CdlE3<<" "<<E0<<" "<<Ru<<" "<<k0<<" "<<alpha<<" "<<Itot0<<" "<<u1n0<<" "<<dt<<" "<<gamma<<" dicts"<<"\n";

    Itot1 = Itot0;
    double Itot_bound =bounds_val;//std::max(10*Cdlp*delta_E*omega/Nt,1.0);

    for (int n_out = 0; n_out < t.size(); n_out++) {
        while (t1 < t[n_out]) {
            Itot0 = Itot1;
            if (input==1){
              E = et(E_start, omega, phase,delta_E ,t1+dt);
              dE = dEdt(omega, cap_phase,delta_E , t1+0.5*dt);
              cap_E=et(E_start, omega, cap_phase,delta_E ,t1+dt);
            }
            else if (input ==0){
              E = c_et( E_start, E_reverse, tr, omega,  phase,  v,  delta_E, t1);
              dE = c_dEdt(tr,  omega,  cap_phase,  v,  delta_E, t1+0.5*dt);
              cap_E= c_et( E_start, E_reverse, tr, omega,  cap_phase,  v,  delta_E, t1);
            }
            else if (input ==2){
              E=dcv_et(E_start, E_reverse,tr,v, t1);
              dE=dcv_dEdt(tr,v, t1+0.5*dt);
              cap_E=E;
            }
            else if (input==3){
              E=sum_of_sinusoids_E(amp_vector, freq_vector, phase_vector, num_frequencies, t1);
              dE=sum_of_sinusoids_dE(amp_vector, freq_vector, phase_vector, num_frequencies, t1);;
              cap_E=E;
            }


            e_surface_fun bc(E,dE,cap_E,Cdl,CdlE,CdlE2,CdlE3,E0,Ru,k0,alpha,Itot0,u1n0,dt,gamma, Upper_lambda, integral_0_oxidation, integral_0_reduction, Marcus_flag);
            boost::uintmax_t max_it = max_iterations;
            //Itot1 = boost::math::tools::newton_raphson_iterate(bc, Itot0,Itot0-Itot_bound,Itot0+Itot_bound, digits_accuracy, max_it);
            std::pair <double, double> sol=boost::math::tools::brent_find_minima(bc,Itot0-Itot_bound,Itot0+Itot_bound, digits_accuracy, max_it);
            //cout.precision(std::numeric_limits<double>::digits10);
            ///if (max_it == max_iterations) throw std::runtime_error("non-linear solve for Itot[n+1] failed, max number of iterations reached");
            Itot1=sol.first;
            bc.update_temporaries(Itot1);
          //  cout<<bc.residual(sol.first)<<"\n";
            if (debug!=-1 && debug<t[n_out]){
              std::vector<vector<double>> diagnostic=NR_function_surface(bc, Itot1, Itot0, Itot_bound);
              cout<<Cdlp*(dt*dE-Ru*(Itot1-Itot0))<<" "<<gamma*(bc.u1n1-bc.u1n0)<<"\n";
              cout<<-Cdlp*Ru<<" "<<  dt <<" "<< gamma*bc.du1n1<<"\n";
              return py::cast(diagnostic);
            }
            theta_0=u1n0;
            u1n0 = bc.u1n1;
            t1 += dt;

        }
        Itot[n_out] =(Itot1-Itot0)*(t[n_out]-t1+dt)/dt + Itot0;
        //Itot[n_out]=u1n0;//(u1n0-theta_0)*(t[n_out]-t1+dt)/dt + theta_0;
    }
    return py::cast(Itot);
}
PYBIND11_MODULE(isolver_martin_brent, m) {
	m.def("brent_current_solver", &brent_current_solver, "solve for I_tot using the brent minimisation method");
  m.def("c_et", &c_et, "Classical potential input");
  m.def("c_dEdt", &c_dEdt, "Classical potential input derivative");
  m.def("et", &et, "Ramp-free potential input");
  m.def("dEdt", &dEdt, "Ramp-free potential derivative");
  m.def("dcv_et", &dcv_et, "DCV potential input");
  m.def("dcv_dEdt", &dcv_dEdt, "DCV potential derivative");
  m.def("sum_of_sinusoids_E", &sum_of_sinusoids_E, "SoS potential");
  m.def("sum_of_sinusoids_dE", &sum_of_sinusoids_dE, "SoS potential deriv");
}
