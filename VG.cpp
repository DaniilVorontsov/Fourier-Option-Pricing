#include"FourierMethods.h"

void VG(bool errorFlag, bool timeFlag)
{
	//initialize strike grid
	double lb = 80;		//left boundary strike
	double rb = 120;	//right boundary strike
	int N_strikes = 50;	//number of strikes
	Double_v K_grid(N_strikes + 1);
	for (int j = 0; j <= N_strikes; j++) K_grid[j] = lb + j * (rb - lb) / N_strikes;


	//SET 1 -- Option Valuation Using the Fast Fourier Transform (1999)
	double S0(100), r(0.05), q(0.03), T(0.25), sigma(0.25), theta(-0.1), nu(2.0);
	VarianceGammaMarket VGoption1(S0, r, q, sigma, theta, nu);

	//SET 2 -- novel option pricing method based on Fourier-cosine series (2008)
	S0 = 100, r = 0.1, q = 0.0, T = 1.0, theta = -0.14, nu = 0.2, sigma = 0.12;
	VarianceGammaMarket VGoption2(S0, r, q, sigma, theta, nu);


	bool CallFlag = true;	//true for call option, false for put option
	Int_v N_terms({ 512, 1024,2048,4096,8192 });	//Number of terms in Carr Madan FFT, COS and FST
	int N_steps = int(1e5);							//Number of steps in Direct integration by Carr Madan Formula

	double alpha(1.5), dk(0.0125); //Carr Madan parameters
	double eps(1e-2); //error in direct Carr Madan integration
	bool splineFlag = true;					 //spline interpolation in FFT and FST

	double L = 8; //integration range in COS method

	///////////////////////////ERROR ANALYSIS///////////////////////////

	if (errorFlag)
	{
		ofstream VG_Errors("VG_Errors.csv"); VG_Errors << "N\tK\tTruePrice\tCM\tFST\tCOS\tSET\n";
		cout << "SET 1\n";
		for (int k = 0; k < N_terms.size(); k++)
		{
			int N = N_terms[k];

			Double_v CMprices1 = VGoption1.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);

			for (int j = 0; j < K_grid.size(); j++)
			{
				double truePrice = VGoption1.PriceByIntegrationVG(K_grid[j], T, CallFlag);
				double FSTprice = VGoption1.PriceByFST(K_grid[j], T, N, CallFlag, splineFlag);
				double COSprice = VGoption1.PriceByCOS(K_grid[j], T, N, L, CallFlag);

				VG_Errors << N << '\t' << K_grid[j] << '\t' << truePrice << '\t' << CMprices1[j] << '\t' << FSTprice << '\t' << COSprice << "\t1\n";
			}

		}
		cout << "Pricing VG SET 1 finished\n";
		cout << "SET 2\n";
		for (size_t k = 0; k < N_terms.size(); k++)
		{
			Double_v CMprices2;
			int N = N_terms[k];
			CMprices2 = VGoption2.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);

			for (int j = 0; j < K_grid.size(); j++)
			{
				double truePrice = VGoption2.PriceByIntegrationVG(K_grid[j], T, CallFlag);
				double FSTprice = VGoption2.PriceByFST(K_grid[j], T, N, CallFlag, splineFlag);
				double COSprice = VGoption2.PriceByCOS(K_grid[j], T, N, L, CallFlag);

				VG_Errors << N << '\t' << K_grid[j] << '\t' << truePrice << '\t' << CMprices2[j] << '\t' << FSTprice << '\t' << COSprice << "\t2\n";
			}
		}
		cout << "Pricing VG SET 2 finished\n";
	}


	///////////////////////////TIME PERFORMANCE///////////////////////////

	if (timeFlag)
	{
		//time measurement
		auto start = high_resolution_clock::now();
		auto stop = high_resolution_clock::now();

		ofstream VG_Time("VG_Time.csv"); VG_Time << "N\tAnalytical\tCM\tFST\tCOS\tSET\n";

		//SET 1
		//measure analytical time separately since it does not depend on N_terms
		long long AnalyticalTime1(1e15);
		for (int t = 0; t < 2; t++)
		{
			start = high_resolution_clock::now();
			for (int j = 0; j < K_grid.size(); j++) double truePrice = VGoption1.PriceByIntegrationVG(K_grid[j], T, CallFlag);
			stop = high_resolution_clock::now();
			AnalyticalTime1 = min(AnalyticalTime1, duration_cast<milliseconds>(stop - start).count());
		}


		for (int n = 0; n < N_terms.size(); n++)
		{

			long long CMtime(1e15), FSTtime(1e15), COStime(1e15);
			int N = N_terms[n];
			for (int t = 0; t < 10; t++)
			{
				start = high_resolution_clock::now();
				Double_v CMprices1 = VGoption1.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);
				stop = high_resolution_clock::now();
				CMtime = min(CMtime, duration_cast<microseconds>(stop - start).count());


				start = high_resolution_clock::now();
				for (int j = 0; j < K_grid.size(); j++) double FSTprice = VGoption1.PriceByFST(K_grid[j], T, N, CallFlag, splineFlag);
				stop = high_resolution_clock::now();
				FSTtime = min(FSTtime, duration_cast<milliseconds>(stop - start).count());

				start = high_resolution_clock::now();
				for (int j = 0; j < K_grid.size(); j++) double COSprice = VGoption1.PriceByCOS(K_grid[j], T, N, L, CallFlag);
				stop = high_resolution_clock::now();
				COStime = min(COStime, duration_cast<milliseconds>(stop - start).count());
			}
			VG_Time << N << '\t' << AnalyticalTime1 << '\t' << double(CMtime) / 1000 << '\t' << FSTtime << '\t' << COStime << "\t1\n";
		}

		cout << "Time measurement VG SET 1 finished\n";



		//SET 2
		//measure analytical time separately since it does not depend on N_terms
		long long AnalyticalTime2(1e15);
		for (int t = 0; t < 10; t++)
		{
			start = high_resolution_clock::now();
			for (int j = 0; j < K_grid.size(); j++) double truePrice = VGoption2.PriceByIntegrationVG(K_grid[j], T, CallFlag);
			stop = high_resolution_clock::now();
			AnalyticalTime2 = min(AnalyticalTime2, duration_cast<milliseconds>(stop - start).count());
		}


		for (int n = 0; n < N_terms.size(); n++)
		{

			long long CMtime(1e15), FSTtime(1e15), COStime(1e15);
			int N = N_terms[n];
			for (int t = 0; t < 10; t++)
			{
				start = high_resolution_clock::now();
				Double_v CMprices1 = VGoption2.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);
				stop = high_resolution_clock::now();
				CMtime = min(CMtime, duration_cast<microseconds>(stop - start).count());


				start = high_resolution_clock::now();
				for (int j = 0; j < K_grid.size(); j++) double FSTprice = VGoption2.PriceByFST(K_grid[j], T, N, CallFlag, splineFlag);
				stop = high_resolution_clock::now();
				FSTtime = min(FSTtime, duration_cast<milliseconds>(stop - start).count());

				start = high_resolution_clock::now();
				for (int j = 0; j < K_grid.size(); j++) double COSprice = VGoption2.PriceByCOS(K_grid[j], T, N, L, CallFlag);
				stop = high_resolution_clock::now();
				COStime = min(COStime, duration_cast<milliseconds>(stop - start).count());
			}
			VG_Time << N << '\t' << AnalyticalTime2 << '\t' << double(CMtime) / 1000 << '\t' << FSTtime << '\t' << COStime << "\t2\n";
		}

		cout << "Time measurement VG SET 2 finished\n\n";
	}

}


