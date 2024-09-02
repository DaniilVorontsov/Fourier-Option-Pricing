#include"FourierMethods.h"

void VG()
{
	//time measurement
	auto start = high_resolution_clock::now();
	auto stop = high_resolution_clock::now();


	//initialize strike grid
<<<<<<< Updated upstream
	double lb = 80;
	double rb = 120;
	int L = 50;
	Double_v K_grid(L + 1);
	for (int j = 0; j <= L; j++) K_grid[j] = lb + j * (rb - lb) / L;
=======
	double lb = 80;		//left boundary strike
	double rb = 120;	//right boundary strike
	int N_strikes = 50;	//number of strikes
	Double_v K_grid(N_strikes + 1);
	for (int j = 0; j <= N_strikes; j++) K_grid[j] = lb + j * (rb - lb) / N_strikes;
>>>>>>> Stashed changes


	//SET 1 -- [1]
	double S0(100), r(0.05), q(0.03), T(0.25), sigma(0.25), theta(-0.1), nu(2.0);
	VarianceGammaMarket VGoption1(S0, r, q, sigma, theta, nu);

	//SET 2 -- [3]
	S0 = 100, r = 0.1, q = 0.0, T = 1.0, theta = -0.14, nu = 0.2, sigma = 0.12;
	VarianceGammaMarket VGoption2(S0, r, q, sigma, theta, nu);


	bool CallFlag = true;	//true for call option, false for put option
	Int_v N_terms({ 512, 1024,2048,4096,8192 });	//Number of terms in Carr Madan FFT, COS and FST
	int N_steps = int(1e5);							//Number of steps in Direct integration by Carr Madan Formula

<<<<<<< Updated upstream
	double alpha(1.5), dk(0.025), eps(1e-2); //Carr Madan parameters
=======
	double alpha(1.5), dk(0.0125); //Carr Madan parameters
	double eps(1e-2); //error in direct Carr Madan integration
>>>>>>> Stashed changes
	bool splineFlag = true;					 //spline interpolation in FFT and FST

	///////////////////////////ERROR ANALYSIS///////////////////////////

	/*ofstream VG_Errors("VG_Errors.csv"); VG_Errors << "N\tK\tTruePrice\tCM\tFST\tCOS\tSET\n";

	for (int k = 0; k < N_terms.size(); k++)
	{
<<<<<<< Updated upstream
		int N = N_terms[k];

		Double_v CMprices1 = VGoption1.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);

		for (int j = 0; j < K_grid.size(); j++)
=======
		ofstream VG_Errors("VG_Errors.csv"); VG_Errors << "N\tK\tTruePrice\tCM\tFST\tCOS\tSET\n";
		cout << "SET 1\n";
		for (int k = 0; k < N_terms.size(); k++)
>>>>>>> Stashed changes
		{
			double truePrice = VGoption1.PriceByIntegrationVG(K_grid[j], T, CallFlag);
			double FSTprice = VGoption1.PriceByFST(K_grid[j], T, N, CallFlag, splineFlag);
			double COSprice = VGoption1.PriceByCOS(K_grid[j], T, N, CallFlag);

			VG_Errors << N << '\t' << K_grid[j] << '\t' << truePrice << '\t' << CMprices1[j] << '\t' << FSTprice << '\t' << COSprice << "\t1\n";
		}
<<<<<<< Updated upstream

=======
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
>>>>>>> Stashed changes
	}
	cout << "VG SET 1 finished\n";

	for (size_t k = 0; k < N_terms.size(); k++)
	{
		Double_v CMprices2;
		int N = N_terms[k];
		CMprices2 = VGoption2.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);

		for (int j = 0; j < K_grid.size(); j++)
		{
			double truePrice = VGoption2.PriceByIntegrationVG(K_grid[j], T, CallFlag);
			double FSTprice = VGoption2.PriceByFST(K_grid[j], T, N, CallFlag, splineFlag);
			double COSprice = VGoption2.PriceByCOS(K_grid[j], T, N, CallFlag);

			VG_Errors << N << '\t' << K_grid[j] << '\t' << truePrice << '\t' << CMprices2[j] << '\t' << FSTprice << '\t' << COSprice << "\t2\n";
		}
	}
	cout << "VG SET 2 finished\n";*/


	///////////////////////////TIME PERFORMANCE///////////////////////////

	ofstream VG_Time("VG_Time.txt"); VG_Time << "N\tAnalytical\tCM\tFST\tCOS\tSET\n";


	for (int n = 0; n < N_terms.size(); n++)
	{

<<<<<<< Updated upstream
		long long AnalyticalTime(1e15), CMtime(1e15), FSTtime(1e15), COStime(1e15);
		int N = N_terms[n];
=======
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
>>>>>>> Stashed changes
		for (int t = 0; t < 10; t++)
		{
			start = high_resolution_clock::now();
			Double_v CMprices1 = VGoption1.PricesByCarrMadanFFT(K_grid, T, dk, alpha, N, CallFlag, splineFlag);
			stop = high_resolution_clock::now();
			CMtime = min(CMtime, duration_cast<microseconds>(stop - start).count());
				

			start = high_resolution_clock::now();
			for (int k = 0; k < K_grid.size(); k++)
			{
				for (int j = 0; j < K_grid.size(); j++) double truePrice = VGoption1.PriceByIntegrationVG(K_grid[k], T, CallFlag);
			}
			stop = high_resolution_clock::now();
			AnalyticalTime = min(AnalyticalTime, duration_cast<milliseconds>(stop - start).count());

<<<<<<< Updated upstream
			start = high_resolution_clock::now();
			for (int k = 0; k < K_grid.size(); k++)
			{
				for (int j = 0; j < K_grid.size(); j++) double FSTprice = VGoption1.PriceByFST(K_grid[k], T, N, CallFlag, splineFlag);
			}
			stop = high_resolution_clock::now();
			FSTtime = min(FSTtime, duration_cast<milliseconds>(stop - start).count());

			start = high_resolution_clock::now();
			for (int k = 0; k < K_grid.size(); k++)
			{
				for (int j = 0; j < K_grid.size(); j++) double COSprice = VGoption1.PriceByCOS(K_grid[k], T, N, CallFlag);
			}
			stop = high_resolution_clock::now();
			COStime = min(COStime, duration_cast<milliseconds>(stop - start).count());

			cout << "Trial " << t + 1 << '\t' << N << endl;
			//cout << AnalyticalTime << '\t' << double(CMtime) / 1000 << '\t' << FSTtime << '\t' << COStime << "\t1\n";
		}
		VG_Time << N << '\t' << AnalyticalTime << '\t' << double(CMtime) / 1000 << '\t' << FSTtime << '\t' << COStime << "\t1\n";
=======
		cout << "Time measurement VG SET 2 finished\n\n";
>>>>>>> Stashed changes
	}


}


