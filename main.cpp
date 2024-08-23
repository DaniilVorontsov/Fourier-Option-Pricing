#include"FourierMethods.h"

void VG()
{
	//SET 1 -- [3]
	//double S0(100), r(0.1), q(0.0), T(1.0), sigma(0.12), theta(-0.14), nu(0.2);

	//SET 2 -- [1]
	double S0(100), r(0.05), q(0.03), T(0.25), theta(-0.10), nu(2.0), sigma(0.25);


	double lb = 80;
	double rb = 120;
	int L = 50;
	Double_v K_grid(L + 1);

	double K = 100;

	for (int j = 0; j <= L; j++) K_grid[j] = lb + j * (rb - lb) / L;


	bool CallFlag = true;
	int N_terms = 2048;		//Number of terms in Carr Madan FFT, COS and FST
	int N_steps = int(1e5); //Number of steps in Direct integration by Carr Madan Formula

	double alpha(1.5), dk(0.025);
	double eps(1e-2);

	VarianceGammaMarket VGoption(S0, r, q, sigma, theta, nu);
	cout << "Variance Gamma\n";
	cout << VGoption.PriceByIntegrationVG(K, T, CallFlag) << "\t Analytical formula" << endl;
	cout << VGoption.PriceByCarrMadanDirect(K, T, alpha, N_steps, eps, CallFlag) << "\t CM direct" << endl;
	cout << VGoption.PriceByFST(K, T, N_terms, CallFlag) << "\t FST" << endl;
	cout << VGoption.PriceByCOS(K, T, N_terms, CallFlag) << "\t COS" << endl;


}


void CGMY()
{
	bool CallFlag = true;
	int N_terms = 2048;		//Number of terms in Carr Madan FFT, COS and FST
	int N_steps = int(1e6); //Number of steps in Direct integration by Carr Madan Formula
	int N_sim = int(1e6);   //Number of simulations for CGMY Monte Carlo


	double alpha(1.5), dk(0.025);
	double eps(1e-2);

	//SET 1 -- Forsyth, Wan, and Wang(2007). Ref price: 4.3714..
	double S0(10), K(10), T(0.25), r(0.1), q(0.0), C(0.22), G(0.75), M(1.0), Y(1.8);
	cout << "SET 1";
	CGMYmarket CGMYoption1(S0, r, q, C, G, M, Y);
	cout << "\nCGMY Call Option\n";
	cout << CGMYoption1.PriceByMonteCarlo(K, T, N_sim, CallFlag) << "\t Monte Carlo" << endl;
	cout << CGMYoption1.PriceByCarrMadanDirect(K, T, alpha, N_steps, eps, CallFlag) << "\t Carr Madan direct" << endl;
	cout << CGMYoption1.PriceByFST(K, T, N_terms, CallFlag) << "\t FST" << endl;
	cout << CGMYoption1.PriceByCOS(K, T, N_terms, CallFlag) << "\t COS" << endl;

	CallFlag = false;
	cout << "\nCGMY Put Option\n";
	cout << CGMYoption1.PriceByMonteCarlo(K, T, N_sim, CallFlag) << "\t Monte Carlo" << endl;
	cout << CGMYoption1.PriceByCarrMadanDirect(K, T, alpha, N_steps, eps, CallFlag) << "\t Carr Madan direct" << endl;
	cout << CGMYoption1.PriceByFST(K, T, N_terms, CallFlag) << "\t FST" << endl;
	cout << CGMYoption1.PriceByCOS(K, T, N_terms, CallFlag) << "\t COS" << endl;


	//SET 2 -- [2] COS method for option pricing (2008). Ref. price: 19.8129..
	S0 = 100, K = 100, T = 1, r = 0.1, q = 0.0, C = 1, G = 5, M = 5, Y = 0.5;
	cout << "\nSET 2";
	CGMYmarket CGMYoption2(S0, r, q, C, G, M, Y);
	CallFlag = true;
	cout << "\nCGMY Call Option\n";
	cout << CGMYoption2.PriceByMonteCarlo(K, T, N_sim, CallFlag) << "\t Monte Carlo" << endl;
	cout << CGMYoption2.PriceByCarrMadanDirect(K, T, alpha, N_steps, eps, CallFlag) << "\t Carr Madan direct" << endl;
	cout << CGMYoption2.PriceByFST(K, T, N_terms, CallFlag) << "\t FST" << endl;
	cout << CGMYoption2.PriceByCOS(K, T, N_terms, CallFlag) << "\t COS" << endl;

	CallFlag = false;
	cout << "\nCGMY Put Option\n";
	cout << CGMYoption2.PriceByMonteCarlo(K, T, N_sim, CallFlag) << "\t Monte Carlo" << endl;
	cout << CGMYoption2.PriceByCarrMadanDirect(K, T, alpha, N_steps, eps, CallFlag) << "\t Carr Madan direct" << endl;
	cout << CGMYoption2.PriceByFST(K, T, N_terms, CallFlag) << "\t FST" << endl;
	cout << CGMYoption2.PriceByCOS(K, T, N_terms, CallFlag) << "\t COS" << endl;




	//SET 3 -- Monte Carlo Simulation of the CGMY Process and Option Pricing. Ref. price: 14.0691..
	S0 = 100, K = 100, T = 1, r = 0.04, q = 0.0, C = 0.5, G = 2.0, M = 3.5, Y = 0.5;
	cout << "\nSET 3";
	CGMYmarket CGMYoption3(S0, r, q, C, G, M, Y);
	CallFlag = true;
	cout << "\nCGMY Call Option\n";
	cout << CGMYoption3.PriceByMonteCarlo(K, T, N_sim, CallFlag) << "\t Monte Carlo" << endl;
	cout << CGMYoption3.PriceByCarrMadanDirect(K, T, alpha, N_steps, eps, CallFlag) << "\t Carr Madan direct" << endl;
	cout << CGMYoption3.PriceByFST(K, T, N_terms, CallFlag) << "\t FST" << endl;
	cout << CGMYoption3.PriceByCOS(K, T, N_terms, CallFlag) << "\t COS" << endl;

	CallFlag = false;
	cout << "\nCGMY Put Option\n";
	cout << CGMYoption3.PriceByMonteCarlo(K, T, N_sim, CallFlag) << "\t Monte Carlo" << endl;
	cout << CGMYoption3.PriceByCarrMadanDirect(K, T, alpha, N_steps, eps, CallFlag) << "\t Carr Madan direct" << endl;
	cout << CGMYoption3.PriceByFST(K, T, N_terms, CallFlag) << "\t FST" << endl;
	cout << CGMYoption3.PriceByCOS(K, T, N_terms, CallFlag) << "\t COS" << endl;


}

void CGMY_MC();




int main()
{

	//VG();
	CGMY();

	//CGMY_MC();

	return 0;
}