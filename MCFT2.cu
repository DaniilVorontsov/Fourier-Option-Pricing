
#include <cuda_runtime.h>
#include <curand_kernel.h>  // Include the CURAND header

#include"FourierMethods.h"



// CUDA kernel function to price CGMY option
__global__ void PriceByMC2(float* price, double S0, double r, double q, double w, double T, double K, int N_sim, double* ST) {
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int totalThreads = blockDim.x * gridDim.x;

	float localPrice = 0;

	for (int i = idx; i < N_sim; i += totalThreads)
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





double MCFT2(double S0, double  K, double  T, double  r, double  C, double  G, double  M, double  Y, double q, double D, int N_sim) {

	////////////////////////////////CDF COMPUTATION////////////////////////////////

	// Number of points to generate and CUDA parameters
	const int threadsPerBlock = 1024;
	const int numBlocks = (N_sim + threadsPerBlock - 1) / threadsPerBlock;

	//CDF Resolution paramters
	const double L = 150;
	const int N_points = 1001; //number of points to recover

	const double eta = D / N_points;
	const double h = L / N_points;

	double* x;
	cudaMallocManaged((void**)&x, N_points * sizeof(double));

	double* u = new double[N_points];
	Complex* phi_r = new Complex[N_points];
	for (int j = 0; j < N_points; j++)
	{
		x[j] = -double(N_points) / 2 * eta + j * eta;
		Complex u_j = -double(N_points) / 2 * h + j * h;
		u[j] = u_j.real();
		Complex phi_r_j = -(1 - cos(u[j] * D)) / (i * u[j]) * CF_CGMY(u[j], T, S0, r, q, C, G, M, Y);
		phi_r[j] = phi_r_j;
	}

	
	double* F_x;
	//Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged((void**)&F_x, N_points * sizeof(double));

	for (int l = 0; l < N_points; l++)
	{
		Complex F_l = 0;
		for (int j = 0; j < N_points; j++)
		{
			F_l = F_l + exp(-i * u[j] * x[l]) * phi_r[j];
		}
		F_x[l] = F_l.real() * h / (2 * pi) + 0.5;
	}
	delete[] u;

	////////////////////////////////SAMPLING OF UNIFORMAL VARIABLES////////////////////////////////

	double* d_uniform;
	cudaMalloc((void**)&d_uniform, N_sim * sizeof(double));

	// Create a device pointer on the host, to hold the random states
	curandState* d_state;
	cudaMalloc((void**)&d_state, N_sim * sizeof(curandState));

	unsigned long seed = time(NULL);
	// Init the random states
	init_rng2 << <numBlocks, threadsPerBlock >> > (seed, d_state);
	// Generate numbers in GPU
	gen_x2 << <numBlocks, threadsPerBlock >> > (d_state, d_uniform);

	////////////////////////////////INTERPOLATING WITH CDF////////////////////////////////

	double* d_ST;
	cudaMallocManaged((void**)&d_ST, N_sim * sizeof(double));

	linearInterpolateKernel2 << <numBlocks, threadsPerBlock >> > (F_x, x, d_uniform, d_ST, N_points, N_sim);

	cudaFree(d_uniform);
	cudaFree(F_x);
	cudaFree(x);

	////////////////////////////////PRICING OPTIONS////////////////////////////////

	float h_price;
	float* d_price;

	cudaMalloc((void**)&d_price, sizeof(float));
	cudaMemset(d_price, 0, sizeof(float));

	double w = -C * tgamma(-Y) * (pow(M - 1, Y) - pow(M, Y) + pow(G + 1, Y) - pow(G, Y));

	PriceByMC2 << <numBlocks, threadsPerBlock >> > (d_price, S0, r, q, w, T, K, N_sim, d_ST);

	cudaMemcpy(&h_price, d_price, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_price);

	cudaDeviceReset();

	return double(h_price);

}

