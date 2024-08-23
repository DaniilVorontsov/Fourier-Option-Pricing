
#include"FourierMethods.h"


Complex CF_CGMY(Complex u, double T, double S0, double r, double q, double C, double G, double M, double Y)
{
	double w = -C * tgamma(-Y) * (pow(M - 1, Y) - pow(M, Y) + pow(G + 1, Y) - pow(G, Y));
	double mu = r + w - q;
	Complex phi_CGMY = exp(C * T * tgamma(-Y) * (pow(M - i * u, Y) - pow(M, Y) + pow(G + i * u, Y) - pow(G, Y)));
	return exp(i * u * (log(S0) + mu * T)) * phi_CGMY;
}

void COS_CGMY_pdf(double* x, double* f, int J, int N, double T, double S0, double r, double q, double C, double G, double M, double Y, double a, double b)
{
	Double_v F;
	for (int k = 0; k < N; k++)
	{
		double w = -C * tgamma(-Y) * (pow(M - 1, Y) - pow(M, Y) + pow(G + 1, Y) - pow(G, Y));
		double mu = r + w - q;
		double u_k = k * pi / (b - a);
		double F_k = 2 / (b - a) * real(CF_CGMY(u_k, T, S0, r, q, C, G, M, Y) * exp(-i * u_k * (log(S0) + mu * T)) * exp(-i * u_k * a));
		F.push_back(F_k);
	}
	F[0] *= 0.5;

	for (int k = 0; k < J; k++)
	{
		double x_k = x[k];
		double f_x = 0;
		for (int k = 0; k < N; k++) {
			double u_k = k * pi / (b - a);
			double costerm = cos(u_k * (x_k - a));
			f_x += F[k] * costerm;
		}
		f[k] = f_x;
	}
}


// CUDA kernel function to price CGMY option
__global__ void PriceByMC(float* price, double S0, double r, double q, double w, double T, double K, int N_sim, double* ST) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int totalThreads = blockDim.x * gridDim.x;

	float localPrice = 0;

	for (int i = tid; i < N_sim; i += totalThreads)
	{
		double S = S0 * exp((r - q + w) * T + ST[i]);
		double payoff = max(K - S, 0.0);
		//localPrice.add(payoff * exp(-r * T));
		localPrice += payoff * exp(-r * T) / N_sim;
	}

	atomicAdd(price, localPrice);
}


// Linear interpolation kernel
__global__ void linearInterpolateKernel(double* d_x, double* d_y, double* d_interp_x, double* d_interp_y, int J, int N_sim) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N_sim) {
		double x = d_interp_x[idx];

		// Find the interval x is in
		for (int i = 0; i < J - 1; i++) {
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
__global__ void init_rng(unsigned int seed, curandState* state)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__global__ void gen_x(curandState* state, double* x)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curandState localState = state[idx];
	x[idx] = curand_uniform_double(&localState);
	state[idx] = localState;
}





void CGMY_MC() {

	////////////////////////////////CDF COMPUTATION////////////////////////////////
	double S0(10), K(10), T(0.25), r(0.1), C(0.22), G(0.75), M(1.0), Y(1.8), q(0.0);

	// Number of points to generate and CUDA parameters
	const int N_sim = 1e7;
	const int threadsPerBlock = 1024;
	const int numBlocks = (N_sim + threadsPerBlock - 1) / threadsPerBlock;

	//COS parameters
	double L = 8;
	const int N_COS_TERMS = int(pow(2, 14));
	double a = -L * sqrt(T);
	double b = L * sqrt(T);

	//size of a pdf/cdf
	int J = 1000;
	double x_min = -4;
	double x_max = 4;
	double dx = (x_max - x_min) / J;


	double* x;
	cudaMallocManaged((void**)&x, J * sizeof(double));
	for (int j = 0; j < J; j++) x[j] = x_min + j * dx;

	double* f_x = new double[J];

	COS_CGMY_pdf(x, f_x, J, N_COS_TERMS, T, S0, r, q, C, G, M, Y, a, b);

	double* F_x;
	cudaMallocManaged((void**)&F_x, J * sizeof(double));

	double F_j = f_x[0];
	for (int j = 0; j < J; j++)
	{
		F_j += (j == 0) ? 0 : f_x[j];
		F_x[j] = F_j * dx;
	}


	delete[] f_x;

	double w = -C * tgamma(-Y) * (pow(M - 1, Y) - pow(M, Y) + pow(G + 1, Y) - pow(G, Y));
	double mu = r + w - q;


	////////////////////////////////SAMPLING OF UNIFORMAL VARIABLES////////////////////////////////
	auto start = high_resolution_clock::now();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);


	double* d_uniform;
	//Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged((void**)&d_uniform, N_sim * sizeof(double));

	// Create a device pointer on the host, to hold the random states
	curandState* d_state;
	cudaMalloc((void**)&d_state, N_sim * sizeof(curandState));

	unsigned long seed = time(NULL);
	// Init the random states
	init_rng << <numBlocks, threadsPerBlock >> > (seed, d_state);
	// Generate numbers in GPU
	gen_x << <numBlocks, threadsPerBlock >> > (d_state, d_uniform);

	////////////////////////////////INTERPOLATING WITH CDF////////////////////////////////

	double* d_ST;
	cudaMallocManaged((void**)&d_ST, N_sim * sizeof(double));

	linearInterpolateKernel << <numBlocks, threadsPerBlock >> > (F_x, x, d_uniform, d_ST, J, N_sim);

	cudaFree(d_uniform);
	cudaFree(F_x);
	cudaFree(x);

	////////////////////////////////PRICING OPTIONS////////////////////////////////

	float h_price;
	float* d_price;

	cudaMalloc((void**)&d_price, sizeof(float));
	cudaMemset(d_price, 0, sizeof(float));

	PriceByMC << <numBlocks, threadsPerBlock >> > (d_price, S0, r, q, w, T, K, N_sim, d_ST);

	cudaMemcpy(&h_price, d_price, sizeof(float), cudaMemcpyDeviceToHost);
	stop = high_resolution_clock::now(); duration = duration_cast<microseconds>(stop - start);
	cudaFree(d_price);

	cout << "GPU price is " << h_price << endl;
	cout << "GPU pricing time: " << duration.count() / 1000 << " milliseconds" << endl;


	start = high_resolution_clock::now();
	CGMYmarket CGMYoption1(S0, r, q, C, G, M, Y);
	double cpuPrice = CGMYoption1.PriceByMonteCarlo(K, T, N_sim, false);
	cout << "CPU price is " << cpuPrice << endl;
	stop = high_resolution_clock::now(); duration = duration_cast<microseconds>(stop - start);
	cout << "CPU pricing time: " << duration.count() / 1000 << " milliseconds" << endl;


	cout << endl << CGMYoption1.PriceByCarrMadanDirect(K, T, 1.5, 2048, 1e-2, false) << endl;
	cout << endl << CGMYoption1.PriceByFST(K, T, 4096, false) << endl;
	cout << endl << CGMYoption1.PriceByCOS(K, T, 2048, false) << endl;
}

