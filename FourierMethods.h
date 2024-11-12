#pragma once
#include<iostream>

#include<vector>
#include<cmath>
#include<complex>

#include<fstream>
#include<chrono>

#include <random>

#include <boost/lambda/lambda.hpp>
#include <boost/math/special_functions/gamma.hpp> //boost library


#include"LinearInterpolation.h"
#include "Spline.h"

using namespace std;
using namespace std::chrono; //to measure performance

const double pi = 4 * atan(1);
const complex<double> i(0.0, 1.0);

typedef complex<double> Complex;
typedef vector<double> Double_v;
typedef vector<Complex> Complex_v;
typedef vector<int> Int_v;


double N(double x);
double BSM_price(double K, double T, double S0, double r, double q, double sigma);
Complex CF_CGMY(Complex u, double T, double S0, double r, double q, double C, double G, double M, double Y);


class LevyMarket
{
public:
	LevyMarket(double S0_, double r_, double q_) : S0(S0_), r(r_), q(q_) {};
	~LevyMarket() {};

	/////Carr Madan
	double PriceByCarrMadanDirect(double K, double T, double alpha, int N, double eps, bool CallFlag); // price option by direct
	Double_v PricesByCarrMadanFFT(Double_v K_grid, double T, double dk, double alpha, int N, bool CallFlag, bool splineFlag);
	double PriceByFST(double K, double T, int N, bool CallFlag, bool splineFlag);
	double PriceByCOS(double K, double T, int N, double L, bool CallFlag);


protected:
	double S0, r, q;
	virtual double w() = 0; //risk adjusted drift
	virtual Complex CF(Complex u, double T, double S0, double r, double q) = 0; //Characteristic Function
	virtual Complex CE(Complex u, double S0, double r, double q) = 0;			//Characteristic Exponent

	/////Carr Madan psi function
	Complex CarrMadanPsi(double v, double alpha, double T);

	/////COS auxiliary functions
	Double_v PsiCoef(double a, double b, double c, double d, int N);
	Double_v ChiCoef(double a, double b, double c, double d, int N);
	Double_v HCoef(double a, double b, int N, bool CallPut);
};


class VarianceGammaMarket : public LevyMarket
{
public:
	VarianceGammaMarket(double S0_, double r_, double q_, double sigma_, double theta_, double nu_) : LevyMarket(S0_, r_, q_), sigma(sigma_), theta(theta_), nu(nu_) {};
	~VarianceGammaMarket() {};
	double PriceByIntegrationVG(double K, double T, bool CallFlag);

private:
	double sigma, theta, nu;
	double w() { return 1 / nu * log(1.0 - theta * nu - sigma * sigma * nu / 2); };

	Complex CF(Complex u, double T, double S0, double r, double q)
	{
		double mu = r + w() - q;
		Complex phi_vg = pow(1.0 - i * u * theta * nu + nu / 2 * pow(sigma * u, 2), -T / nu);
		return exp(i * u * (log(S0) + mu * T)) * phi_vg;
	}

	Complex CE(Complex u, double S0, double r, double q)
	{
		double mu = r + w() - q;
		return i * u * mu - 1 / nu * log(1.0 - i * theta * nu * u + nu / 2 * pow(sigma * u, 2));
	}

};


class CGMYmarket : public LevyMarket
{
public:
	CGMYmarket(double S0_, double r_, double q_, double C_, double G_, double M_, double Y_) : LevyMarket(S0_, r_, q_), C(C_), G(G_), M(M_), Y(Y_) {};
	~CGMYmarket() {};
	double PriceByMonteCarlo(double K, double T, int N_sim, double D, bool CallFlag);

private:
	double C, G, M, Y;
	Double_v CGMYdensityByCOS(Double_v x, int N, double T, double a, double b);
	double w() { return -C * tgamma(-Y) * (pow(M - 1, Y) - pow(M, Y) + pow(G + 1, Y) - pow(G, Y)); };

	Complex CF(Complex u, double T, double S0, double r, double q)
	{
		double mu = r + w() - q;
		Complex phi_CGMY = exp(C * T * tgamma(-Y) * (pow(M - i * u, Y) - pow(M, Y) + pow(G + i * u, Y) - pow(G, Y)));
		return exp(i * u * (log(S0) + mu * T)) * phi_CGMY;
	}

	Complex CE(Complex u, double S0, double r, double q)
	{
		double mu = r + w() - q;
		return i * u * mu + C * tgamma(-Y) * (pow(M - i * u, Y) - pow(M, Y) + pow(G + i * u, Y) - pow(G, Y));
	}
};
