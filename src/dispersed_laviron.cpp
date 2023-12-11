#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include<map>
#include<string>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
namespace py = pybind11;
using namespace std;
typedef complex<double> dcomp;
template <typename V>
V get(py::dict m, const std::string &key, const V &defval) {
    return m[key.c_str()].cast<V>();
}
std::pair <double, double> calculate_ladder_params(double k0, double e0, double alpha, double gamma, double area, double dc_pot, double F, double R, double T, double FRT){
    double ratio=std::exp(FRT*(e0-dc_pot));
    double ox=gamma/(ratio+1);
    double red=gamma-ox;
    double Ra_coeff=(R*T)/((pow(F,2))*area*k0);
    double nu_1_alpha=std::exp((1-alpha)*FRT*(dc_pot-e0));
    double nu_alpha=std::exp((-alpha)*FRT*(dc_pot-e0));
    double Ra=Ra_coeff*pow((alpha*ox*nu_alpha)+((1-alpha)*red*nu_1_alpha),-1);
    double sigma=k0*Ra*(nu_alpha+nu_1_alpha);
    double Cf=1/sigma;
    std::pair <double, double> return_arg(Ra, Cf);
    return return_arg;
}
py::array Laviron_ladder(py::dict& params, std::vector<double>& frequencies, std::vector<double>& weights, py::array_t<double>& values, std::vector<std::string>& dispersion_parameters){
    py::buffer_info buf_info = values.request();
     
    const double *ptr = static_cast<double *>(buf_info.ptr);
    const size_t rows=static_cast<size_t>(buf_info.shape[0]);
    const size_t cols=static_cast<size_t>(buf_info.shape[1]);
    double F=96485.3321;
    double R=8.3145;
    double T=get(params,std::string("T"),0.0);
    double FRT=F/(R*T);
    map<string, double> current_values;
    current_values["k0"]= get(params,std::string("k_0"),0.0);
    current_values["E0"]= get(params,std::string("E_0"),0.0);
    current_values["alpha"]= get(params,std::string("alpha"),0.1);
    

    double gamma=get(params,std::string("gamma"),0.1);
    double area=get(params,std::string("area"),0.1);
    double dc_pot=get(params,std::string("dc_pot"),0.1);
    dcomp Ru(get(params,std::string("Ru"),0.0), 0.0);
    dcomp Cdl(get(params,std::string("Cdl"),0.0), 0.0);
    dcomp cpe_alpha_cdl(get(params,std::string("cpe_alpha_cdl"),0.0), 0.0);
    std::vector<double> resistors(rows,0);
    std::vector<double> capacitors(rows,0);
    pair<double, double> currentRC;
    
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i * cols + j;
            current_values[dispersion_parameters[j]]=ptr[index];
            
           
        }
        
        currentRC=calculate_ladder_params(current_values["k0"],current_values["E0"], current_values["alpha"], area, gamma, dc_pot, F, R, T, FRT);
        resistors[i]=currentRC.first;
        capacitors[i]=currentRC.second;
        
    }
   
    
    size_t num_freqs=frequencies.size();
    std::vector<std::complex<double>>  impedance(num_freqs,0);
    size_t num_weights=weights.size();
    dcomp z_2;
    dcomp I(0.0,1.0);
    dcomp one(1.0, 0.0);
    
    for (size_t i = 0; i < num_freqs; ++i) {
        dcomp freq(frequencies[i], 0.0);
        
        z_2=Cdl*(pow(I*freq, cpe_alpha_cdl));
        
        for (size_t j=0; j<num_weights; j++){
           
            
            dcomp RN(resistors[j]/weights[j], 0.0);
            dcomp CN(capacitors[j]*weights[j], 0.0);
            
            z_2+=one/(RN+(one/(CN*I*freq)));

            
        }
        impedance[i]=Ru+(one/z_2);
    }
    /*
    result.resize(2, std::vector<double> (num_freqs));
     for (size_t i = 0; i < num_freqs; ++i) {
        result[0][i] = impedance[i].real();    // Real part
        result[1][i] = impedance[i].imag(); // Imaginary part
    }
    */
    constexpr size_t elsize = sizeof(double);
    //size_t shape[2]{2, num_freqs};
    //size_t strides[2]{2 * elsize, elsize};
    py::array_t<double, py::array::c_style> result({2*elsize, num_freqs});
    auto view = result.mutable_unchecked<2>();

    
    for (size_t i = 0; i < num_freqs; ++i) {
        view(0,i) = impedance[i].real();    // Real part
        view(1,i) = impedance[i].imag(); // Imaginary part
    }
    

    return result;
}

    
PYBIND11_MODULE(ladder_laviron, m) {
    m.def("Laviron_ladder", &Laviron_ladder, "quicksolve for Lavironladder");

}