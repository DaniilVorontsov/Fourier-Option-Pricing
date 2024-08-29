#include"FourierMethods.h"

double CGMY_COS_CUDA(double S0, double  K, double  T, double  r, double  C, double  G, double  M, double  Y, double q, double L, int N_sim);
double CGMY_CDF_CUDA(double S0, double  K, double  T, double  r, double  C, double  G, double  M, double  Y, double q, double D, int N_sim);


void CGMY()
{
	bool CallFlag = false; bool splineFlag = false;
	int N_steps = int(1e5); //Number of steps in Direct integration by Carr Madan Formula
	int N_sim = int(1e6);   //Number of simulations for CGMY Monte Carlo
	int N_true_price = int(pow(2, 14));
	double D = 10; //range for CDF recovery in MCFT1 and MCFT2

	double alpha(1.5), dk(0.025);
	double eps(1e-2);

	//initialize strike grid
	double lb = 80;
	double rb = 120;
	int L = 50;
	Double_v K_grid(L + 1);
	for (int j = 0; j <= L; j++) K_grid[j] = lb + j * (rb - lb) / L;

	Int_v N_terms({ 512, 1024,2048,4096,8192 });	//Number of terms in Carr Madan FFT, COS and FST


	///////////////////////////ERROR ANALYSIS///////////////////////////

	ofstream CGMY_Errors("CGMY_Errors.csv"); CGMY_Errors << "N\tK\tTruePrice\tCM\tFST\tCOS\tMCFT1\tMCFT2\tSET\n";


	//////////////SET 1////////////// -- Robust Numerical Valuation of CGMY options (2007). Ref price: 4.3714..
	cout << "SET 1\n";
	double S0(10), K(10), T(0.25), r(0.1), q(0.0), C(1.0), G(8.8), M(9.2), Y(1.8);
	CGMYmarket CGMYoption1(S0, r, q, C, G, M, Y);

	Double_v K_grid_10(L + 1); for (int j = 0; j <= L; j++) K_grid_10[j] = 1 + j * (50 - 1) / L; //strike grid from 1 to 50

	Double_v COS_CUDA(K_grid_10.size()), CDF_CUDA(K_grid_10.size());
	for (int k = 0; k < K_grid_10.size(); k++)
	{
		COS_CUDA[k] = CGMY_COS_CUDA(S0, K_grid_10[k], T, r, C, G, M, Y, q, D/2, N_sim);
		CDF_CUDA[k] = CGMY_CDF_CUDA(S0, K_grid_10[k], T, r, C, G, M, Y, q, D, N_sim);
	}

	for (int k = 0; k < N_terms.size(); k++)
	{
		int N = N_terms[k];

		Double_v CMprices1 = CGMYoption1.PricesByCarrMadanFFT(K_grid_10, T, dk, alpha, N, CallFlag, splineFlag);

		for (int j = 0; j < K_grid_10.size(); j++)
		{
			double truePrice = CGMYoption1.PriceByCOS(K_grid_10[j], T, N_true_price, 2 * D, CallFlag);
			double FSTprice = CGMYoption1.PriceByFST(K_grid_10[j], T, N, CallFlag, splineFlag);
			double COSprice = CGMYoption1.PriceByCOS(K_grid_10[j], T, N, D, CallFlag);

			CGMY_Errors << N << '\t' << K_grid_10[j] << '\t' << truePrice << '\t' << CMprices1[j] << '\t' << FSTprice << '\t' << COSprice << '\t' << COS_CUDA[j] << '\t' << CDF_CUDA[j] << "\t1\n";
		}
	}
	cout << "Pricing CGMY SET 1 finished\n";



	//////////////SET 2////////////// -- COS method for option pricing (2008). Ref. price: 19.8129..
	D = 10;

	S0 = 100, K = 100, T = 1, r = 0.1, q = 0.0, C = 1, G = 5, M = 5, Y = 0.5;
	cout << "\nSET 2\n";
	CGMYmarket CGMYoption2(S0, r, q, C, G, M, Y);

	for (int k = 0; k < K_grid.size(); k++)
	{
		//cout << "MC pricing strike " << K_grid[k] << endl;
		COS_CUDA[k] = CGMY_COS_CUDA(S0, K_grid[k], T, r, C, G, M, Y, q, D, N_sim);
		CDF_CUDA[k] = CGMY_CDF_CUDA(S0, K_grid[k], T, r, C, G, M, Y, q, D, N_sim);
	}

	for (size_t k = 0; k < N_terms.size(); k++)
	{
		Double_v CMprices2;
		int N = N_terms[k];
		CMprices2 = CGMYoption2.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);

		for (int j = 0; j < K_grid.size(); j++)
		{
			double truePrice = CGMYoption2.PriceByCOS(K_grid[j], T, N_true_price, D, CallFlag);
			double FSTprice = CGMYoption2.PriceByFST(K_grid[j], T, N, CallFlag, splineFlag);
			double COSprice = CGMYoption2.PriceByCOS(K_grid[j], T, N, D, CallFlag);

			CGMY_Errors << N << '\t' << K_grid[j] << '\t' << truePrice << '\t' << CMprices2[j] << '\t' << FSTprice << '\t' << COSprice << '\t' << COS_CUDA[j] << '\t' << CDF_CUDA[j] << "\t2\n";
		}
	}
	cout << "Pricing CGMY SET 2 finished\n";


	//////////////SET 3////////////// -- Monte Carlo Simulation of the CGMY Process and Option Pricing. (2013) Ref. price: 14.0691..
	S0 = 100, K = 100, T = 1, r = 0.04, q = 0.0, C = 0.5, G = 2.0, M = 3.5, Y = 0.5;
	cout << "\nSET 3\n";
	CGMYmarket CGMYoption3(S0, r, q, C, G, M, Y);

	for (int k = 0; k < K_grid.size(); k++)
	{
		COS_CUDA[k] = CGMY_COS_CUDA(S0, K_grid[k], T, r, C, G, M, Y, q, D, N_sim);
		CDF_CUDA[k] = CGMY_CDF_CUDA(S0, K_grid[k], T, r, C, G, M, Y, q, D, N_sim);
	}
	for (size_t k = 0; k < N_terms.size(); k++)
	{
		Double_v CMprices2;
		int N = N_terms[k];
		CMprices2 = CGMYoption3.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);

		for (int j = 0; j < K_grid.size(); j++)
		{
			double truePrice = CGMYoption3.PriceByCOS(K_grid[j], T, N_true_price, D, CallFlag);
			double FSTprice = CGMYoption3.PriceByFST(K_grid[j], T, N, CallFlag, splineFlag);
			double COSprice = CGMYoption3.PriceByCOS(K_grid[j], T, N, D, CallFlag);

			CGMY_Errors << N << '\t' << K_grid[j] << '\t' << truePrice << '\t' << CMprices2[j] << '\t' << FSTprice << '\t' << COSprice << '\t' << COS_CUDA[j] << '\t' << CDF_CUDA[j] << "\t3\n";
		}
	}
	cout << "Pricing CGMY SET 3 finished\n";



	///////////////////////////TIME PERFORMANCE///////////////////////////

	//time measurement
	auto start = high_resolution_clock::now();
	auto stop = high_resolution_clock::now();

	//ofstream CGMY_Time("CGMY_Time.csv"); CGMY_Time << "N\tCOSextended\tCM\tFST\tCOS\tMCFT1\tMCFT2\tSET\n";


	//////////////SET1//////////////
	//S0 = 10, K = 10, T = 0.25, r = 0.1, q = 0.0, C = 1, G = 8.8, M = 9.2, Y = 1.8;
	//cout << "\nSET 1\n";
	////measure Monte Carlo time separately since it does not depend on N_terms
	//long long cpuMonteCarlotime1(1e15), gpu1MonteCarlotime1(1e15), gpu2MonteCarlotime1(1e15);
	//for (int t = 0; t < 10; t++)
	//{
	//	start = high_resolution_clock::now();
	//	double truePrice = CGMYoption1.PriceByMonteCarlo(K, T, N_sim, D, CallFlag);
	//	stop = high_resolution_clock::now();
	//	cpuMonteCarlotime1 = min(cpuMonteCarlotime1, duration_cast<milliseconds>(stop - start).count());

	//	start = high_resolution_clock::now();
	//	double MCFT1price = CGMY_COS_CUDA(S0, K, T, r, C, G, M, Y, q, D, N_sim);
	//	stop = high_resolution_clock::now();
	//	gpu1MonteCarlotime1 = min(gpu1MonteCarlotime1, duration_cast<milliseconds>(stop - start).count());

	//	start = high_resolution_clock::now();
	//	double MCFT2price = CGMY_CDF_CUDA(S0, K, T, r, C, G, M, Y, q, D, N_sim);
	//	stop = high_resolution_clock::now();
	//	gpu2MonteCarlotime1 = min(gpu2MonteCarlotime1, duration_cast<milliseconds>(stop - start).count());
	//}


	//for (int n = 0; n < N_terms.size(); n++)
	//{
	//	long long CMtime(1e15), FSTtime(1e15), COStime(1e15);
	//	int N = N_terms[n];
	//	for (int t = 0; t < 10; t++)
	//	{
	//		start = high_resolution_clock::now();
	//		Double_v CMprices1 = CGMYoption1.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);
	//		stop = high_resolution_clock::now();
	//		CMtime = min(CMtime, duration_cast<microseconds>(stop - start).count());


	//		start = high_resolution_clock::now();
	//		for (int k = 0; k < K_grid.size(); k++) double FSTprice = CGMYoption1.PriceByFST(K_grid[k], T, N, CallFlag, splineFlag);
	//		stop = high_resolution_clock::now();
	//		FSTtime = min(FSTtime, duration_cast<milliseconds>(stop - start).count());

	//		start = high_resolution_clock::now();
	//		for (int k = 0; k < K_grid.size(); k++) double COSprice = CGMYoption1.PriceByCOS(K_grid[k], T, N, D, CallFlag);

	//		stop = high_resolution_clock::now();
	//		COStime = min(COStime, duration_cast<milliseconds>(stop - start).count());
	//	}
	//	CGMY_Time << N << '\t' << cpuMonteCarlotime1 * 50 << '\t' << double(CMtime) / 1000 << '\t' << FSTtime << '\t' << COStime << '\t' << gpu1MonteCarlotime1 * 50 << '\t' << gpu2MonteCarlotime1 * 50 << "\t1\n";
	//}
	//cout << "Time measurement CGMY SET 1 finished\n";


	////////////////SET2//////////////
	//S0 = 100, K = 100, T = 1, r = 0.1, q = 0.0, C = 1, G = 5, M = 5, Y = 0.5;
	//cout << "\nSET 2\n";
	////measure Monte Carlo time separately since it does not depend on N_terms
	//long long cpuMonteCarlotime2(1e15), gpu1MonteCarlotime2(1e15), gpu2MonteCarlotime2(1e15);
	//for (int t = 0; t < 10; t++)
	//{
	//	start = high_resolution_clock::now();
	//	double truePrice = CGMYoption1.PriceByMonteCarlo(K, T, N_sim, D, CallFlag);
	//	stop = high_resolution_clock::now();
	//	cpuMonteCarlotime2 = min(cpuMonteCarlotime2, duration_cast<milliseconds>(stop - start).count());

	//	start = high_resolution_clock::now();
	//	double MCFT1price = CGMY_COS_CUDA(S0, K, T, r, C, G, M, Y, q, D, N_sim);
	//	stop = high_resolution_clock::now();
	//	gpu1MonteCarlotime2 = min(gpu1MonteCarlotime2, duration_cast<milliseconds>(stop - start).count());

	//	start = high_resolution_clock::now();
	//	double MCFT2price = CGMY_CDF_CUDA(S0, K, T, r, C, G, M, Y, q, D, N_sim);
	//	stop = high_resolution_clock::now();
	//	gpu2MonteCarlotime2 = min(gpu2MonteCarlotime2, duration_cast<milliseconds>(stop - start).count());
	//}


	//for (int n = 0; n < N_terms.size(); n++)
	//{

	//	long long CMtime(1e15), FSTtime(1e15), COStime(1e15);
	//	int N = N_terms[n];
	//	for (int t = 0; t < 20; t++)
	//	{
	//		start = high_resolution_clock::now();
	//		Double_v CMprices2 = CGMYoption2.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);
	//		stop = high_resolution_clock::now();
	//		CMtime = min(CMtime, duration_cast<microseconds>(stop - start).count());


	//		start = high_resolution_clock::now();
	//		for (int k = 0; k < K_grid.size(); k++) double FSTprice = CGMYoption2.PriceByFST(K_grid[k], T, N, CallFlag, splineFlag);
	//		stop = high_resolution_clock::now();
	//		FSTtime = min(FSTtime, duration_cast<milliseconds>(stop - start).count());

	//		start = high_resolution_clock::now();
	//		for (int k = 0; k < K_grid.size(); k++) double COSprice = CGMYoption2.PriceByCOS(K_grid[k], T, N, D, CallFlag);
	//		stop = high_resolution_clock::now();
	//		COStime = min(COStime, duration_cast<milliseconds>(stop - start).count());
	//	}
	//	CGMY_Time << N << '\t' << cpuMonteCarlotime2 * 50 << '\t' << double(CMtime) / 1000 << '\t' << FSTtime << '\t' << COStime << '\t' << gpu1MonteCarlotime2 * 50 << '\t' << gpu2MonteCarlotime2 * 50 << "\t2\n";
	//}
	//cout << "Time measurement CGMY SET 2 finished\n";


	////////////////SET3//////////////
	//S0 = 100, K = 100, T = 1, r = 0.04, q = 0.0, C = 0.5, G = 2.0, M = 3.5, Y = 0.5;
	//cout << "\nSET 3\n";
	////measure Monte Carlo time separately since it does not depend on N_terms
	//long long cpuMonteCarlotime3(1e15), gpu1MonteCarlotime3(1e15), gpu2MonteCarlotime3(1e15);
	//for (int t = 0; t < 10; t++)
	//{
	//	start = high_resolution_clock::now();
	//	double truePrice = CGMYoption1.PriceByMonteCarlo(K, T, N_sim, D, CallFlag);
	//	stop = high_resolution_clock::now();
	//	cpuMonteCarlotime3 = min(cpuMonteCarlotime3, duration_cast<milliseconds>(stop - start).count());

	//	start = high_resolution_clock::now();
	//	double MCFT1price = CGMY_COS_CUDA(S0, K, T, r, C, G, M, Y, q, D, N_sim);
	//	stop = high_resolution_clock::now();
	//	gpu1MonteCarlotime3 = min(gpu1MonteCarlotime3, duration_cast<milliseconds>(stop - start).count());

	//	start = high_resolution_clock::now();
	//	double MCFT2price = CGMY_CDF_CUDA(S0, K, T, r, C, G, M, Y, q, D, N_sim);
	//	stop = high_resolution_clock::now();
	//	gpu2MonteCarlotime3 = min(gpu2MonteCarlotime3, duration_cast<milliseconds>(stop - start).count());
	//}


	//for (int n = 0; n < N_terms.size(); n++)
	//{

	//	long long CMtime(1e15), FSTtime(1e15), COStime(1e15);
	//	int N = N_terms[n];
	//	for (int t = 0; t < 10; t++)
	//	{
	//		start = high_resolution_clock::now();
	//		Double_v CMprices3 = CGMYoption3.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);
	//		stop = high_resolution_clock::now();
	//		CMtime = min(CMtime, duration_cast<microseconds>(stop - start).count());

	//		start = high_resolution_clock::now();
	//		for (int k = 0; k < K_grid.size(); k++) double FSTprice = CGMYoption3.PriceByFST(K_grid[k], T, N, CallFlag, splineFlag);
	//		stop = high_resolution_clock::now();
	//		FSTtime = min(FSTtime, duration_cast<milliseconds>(stop - start).count());

	//		start = high_resolution_clock::now();
	//		for (int k = 0; k < K_grid.size(); k++) double COSprice = CGMYoption3.PriceByCOS(K_grid[k], T, N, D, CallFlag);
	//		stop = high_resolution_clock::now();
	//		COStime = min(COStime, duration_cast<milliseconds>(stop - start).count());
	//	}
	//	CGMY_Time << N << '\t' << cpuMonteCarlotime3 * 50 << '\t' << double(CMtime) / 1000 << '\t' << FSTtime << '\t' << COStime << '\t' << gpu1MonteCarlotime3 * 50 << '\t' << gpu2MonteCarlotime3 * 50 << "\t3\n";
	//}
	//cout << "Time measurement CGMY SET 3 finished\n";

}


