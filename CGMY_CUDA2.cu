﻿
#include <cuda_runtime.h>
#include <curand_kernel.h>  // Include the CURAND header

#include"FourierMethods.h"



// CUDA kernel function to price CGMY option
__global__ void PriceByMC2(float* price, double S0, double r, double q, double w, double T, double K, int N_sim, double* ST) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int totalThreads = blockDim.x * gridDim.x;

	float localPrice = 0;

	for (int i = tid; i < N_sim; i += totalThreads)
	{
		double S = S0 * exp((r - q + w) * T + ST[i]);
		double payoff = max(K - S, 0.0);
		localPrice += payoff * exp(-r * T) / N_sim;
	}

	atomicAdd(price, localPrice);
}


// Linear interpolation kernel
__global__ void linearInterpolateKernel2(double* d_x, double* d_y, double* d_interp_x, double* d_interp_y, int J, int N_sim) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N_sim) {
		double x = d_interp_x[idx];

		// Find the interval x is in
		for (int i = 0; i < J - 1; i++) {
			//in case of out of boundaries assign closest value
			if (x < d_x[i]) { d_interp_y[idx] = d_y[i]; break; }
			if (x > d_x[J - 1]) { d_interp_y[idx] = d_y[J - 1]; break; }
			if (x >= d_x[i] && x <= d_x[i + 1]) {
				double x0 = d_x[i];
				double x1 = d_x[i + 1];
				double y0 = d_y[i];
				double y1 = d_y[i + 1];

				// Perform linear interpolation
				d_interp_y[idx] = y0 + (x - x0) / (x1 - x0) * (y1 - y0);
				break;
			}
		}
	}
}


//Give a randState to each CUDA thread from which it can sample from
__global__ void init_rng2(unsigned int seed, curandState* state)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__global__ void gen_x2(curandState* state, double* x)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curandState localState = state[idx];
	x[idx] = curand_uniform_double(&localState);
	state[idx] = localState;
}





double CGMY_CUDA2(double S0, double  K, double  T, double  r, double  C, double  G, double  M, double  Y, double q) {

	////////////////////////////////CDF COMPUTATION////////////////////////////////

	// Number of points to generate and CUDA parameters
	const int N_sim = 1e6;
	const int threadsPerBlock = 1024;
	const int numBlocks = (N_sim + threadsPerBlock - 1) / threadsPerBlock;

	//Resolution paramters
	const double L = 150;
	const double D = 10;
	const int N_points = 5000;

	const double eta = D / N_points;
	const double h = L / N_points;

	double* h_x = new double[N_points];
	for (int j = 0; j < N_points; j++) h_x[j] = -double(N_points) / 2 * eta + j * eta;

	//double* f_x = new double[J];

	//COS_CGMY_pdf(h_x, f_x, J, N_COS_TERMS, T, S0, r, q, C, G, M, Y, a, b);

	//int J_size = 0;
	//double F_j = f_x[0];
	//for (int j = 0; j < J; j++)
	//{
	//	F_j += (j == 0) ? 0 : f_x[j] * dx;
	//	if (0.05 <= F_j && F_j <= 0.95) J_size += 1;
	//}

	//double* F_x;
	//cudaMallocManaged((void**)&F_x, J_size * sizeof(double));
	//double* x;
	//cudaMallocManaged((void**)&x, J_size * sizeof(double));

	//F_j = f_x[0]; int k = 0;
	//for (int j = 0; j < J; j++)
	//{
	//	F_j += (j == 0) ? 0 : f_x[j] * dx;
	//	if (0.05 <= F_j && F_j <= 0.95)
	//	{
	//		F_x[k] = F_j;
	//		x[k] = h_x[j];
	//		k++;
	//	}
	//		//cout << f_x[j] <<"\t" << h_x[j] << endl;
	//}



	//delete[] f_x;

	//double w = -C * tgamma(-Y) * (pow(M - 1, Y) - pow(M, Y) + pow(G + 1, Y) - pow(G, Y));
	//double mu = r + w - q;


	//////////////////////////////////SAMPLING OF UNIFORMAL VARIABLES////////////////////////////////
	//auto start = high_resolution_clock::now();
	//auto stop = high_resolution_clock::now();
	//auto duration = duration_cast<microseconds>(stop - start);


	//double* d_uniform;
	////Allocate Unified Memory – accessible from CPU or GPU
	//cudaMalloc((void**)&d_uniform, N_sim * sizeof(double));

	//// Create a device pointer on the host, to hold the random states
	//curandState* d_state;
	//cudaMalloc((void**)&d_state, N_sim * sizeof(curandState));

	//unsigned long seed = time(NULL);
	//// Init the random states
	//init_rng2 << <numBlocks, threadsPerBlock >> > (seed, d_state);
	//// Generate numbers in GPU
	//gen_x << <numBlocks, threadsPerBlock >> > (d_state, d_uniform);

	//////////////////////////////////INTERPOLATING WITH CDF////////////////////////////////

	//double* d_ST;
	//cudaMallocManaged((void**)&d_ST, N_sim * sizeof(double));

	//linearInterpolateKernel2 << <numBlocks, threadsPerBlock >> > (F_x, x, d_uniform, d_ST, J_size, N_sim);

	//cudaFree(d_uniform);
	//cudaFree(F_x);
	//cudaFree(x);

	//////////////////////////////////PRICING OPTIONS////////////////////////////////

	//float h_price;
	//float* d_price;

	//cudaMalloc((void**)&d_price, sizeof(float));
	//cudaMemset(d_price, 0, sizeof(float));

	//PriceByMC2 << <numBlocks, threadsPerBlock >> > (d_price, S0, r, q, w, T, K, N_sim, d_ST);

	//cudaMemcpy(&h_price, d_price, sizeof(float), cudaMemcpyDeviceToHost);
	//stop = high_resolution_clock::now(); duration = duration_cast<microseconds>(stop - start);
	//cudaFree(d_price);

	//cout << "GPU price is " << h_price << endl;
	//cout << "GPU pricing time: " << duration.count() / 1000 << " milliseconds" << endl;


	//start = high_resolution_clock::now();
	//CGMYmarket CGMYoption1(S0, r, q, C, G, M, Y);
	//double cpuPrice = CGMYoption1.PriceByMonteCarlo(K, T, N_sim, false);
	//cout << "CPU price is " << cpuPrice << endl;
	//stop = high_resolution_clock::now(); duration = duration_cast<microseconds>(stop - start);
	//cout << "CPU pricing time: " << duration.count() / 1000 << " milliseconds" << endl;

	return 0;

}
