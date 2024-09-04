
#include"FourierMethods.h"
#include"FFT.h"



///////////////////////////////////BSM PRICE///////////////////////////////////
double BSM_price(double K, double T, double S0, double r, double q, double sigma)
{
	double d1 = (log(S0 / K) + (r - q + 0.5 * pow(sigma, 2.0)) * T) / (sigma * sqrt(T));
	double d2 = d1 - sigma * sqrt(T);
	return S0 * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2);
}

double N(double x)
{
	double gamma = 0.2316419;     double a1 = 0.319381530;
	double a2 = -0.356563782;   double a3 = 1.781477937;
	double a4 = -1.821255978;   double a5 = 1.330274429;
	double k = 1.0 / (1.0 + gamma * x);
	if (x >= 0.0)
	{
		return 1.0 - ((((a5 * k + a4) * k + a3) * k + a2) * k + a1)
			* k * exp(-x * x / 2.0) / sqrt(2.0 * pi);
	}
	else return 1.0 - N(-x);
}


Complex CF_CGMY(Complex u, double T, double S0, double r, double q, double C, double G, double M, double Y)
{
	double w = -C * tgamma(-Y) * (pow(M - 1, Y) - pow(M, Y) + pow(G + 1, Y) - pow(G, Y));
	double mu = r + w - q;
	Complex phi_CGMY = exp(C * T * tgamma(-Y) * (pow(M - i * u, Y) - pow(M, Y) + pow(G + i * u, Y) - pow(G, Y)));
	return /*exp(i * u * (log(S0) + mu * T)) * */phi_CGMY;
}

///////////////////////////CARR MADAN FORMULA///////////////////////////
Complex LevyMarket::CarrMadanPsi(double v, double alpha, double T)
{
	Complex up = exp(-r * T) * CF(v - (alpha + 1) * i, T, S0, r, q);
	Complex down = pow(alpha, 2) + alpha - pow(v, 2) + i * (2 * alpha + 1) * v;
	return up / down;
}

double LevyMarket::PriceByCarrMadanDirect(double K, double T, double alpha, int N, double eps, bool CallFlag)
{
	Complex c = exp(-r * T) * CF(-(alpha + 1) * i, T, S0, r, q);
	double A = exp(-alpha * log(K));
	A *= c.real() / pi / eps;
	double dv = A / N;

	double price = 0;
	for (int j = 1; j <= N; j++)
	{
		double v = dv * (j - 1);
		double val = (exp(-i * v * log(K)) * CarrMadanPsi(v, alpha, T)).real();
		val *= 3 + ((j % 2 == 0) ? 1 : -1) - ((j == 1) ? 1 : 0);
		price += val;
	}
	price *= dv / 3;
	price -= (exp(-i * (A - dv) * log(K)) * CarrMadanPsi((A - dv), alpha, T)).real()
		+ 4 * (exp(-i * A * log(K)) * CarrMadanPsi(A, alpha, T)).real();

	price *= exp(-alpha * log(K)) / pi;
	return (CallFlag) ? price : price + K * exp(-r * T) - S0 * exp(-q * T);
}

Double_v LevyMarket::PricesByCarrMadanFFT(Double_v K_grid, double T, double dk, double alpha, int N, bool CallFlag, bool splineFlag)
{
	double dv = 2 * pi / (N * dk);
	double v_max = dv * N;
	//choose b value so k grid is centered in ATM strike
	double b = dk * N / 2 - log(S0);

	//grid of values in Fourier space
	Double_v v_grid(N);
	for (int j = 0; j < N; j++)
	{
		v_grid[j] = dv * j;
	}

	//values to be passed in FFT
	Complex_v x_fft(N);
	for (int j = 0; j < N; j++)
	{
		double v = v_grid[j];
		Complex up = exp(-r * T) * CF(v - (alpha + 1) * i, T, S0, r, q);
		Complex down = pow(alpha, 2) + alpha - pow(v, 2) + i * (2 * alpha + 1) * v;
		Complex CarrMadanPsi = up / down;
		x_fft[j] = exp(i * b * v_grid[j]) * CarrMadanPsi;
		x_fft[j] *= 3 + (((j + 1) % 2 == 0) ? 1 : -1) - ((j == 0) ? 1 : 0);
		x_fft[j] *= dv / 3;
	}

	//grid of log strikes near ATM values
	Double_v k_grid_atm(N);
	for (int j = 0; j < N; j++)
	{
		k_grid_atm[j] = -b + dk * j;
	}

	FFT(x_fft);

	LinearInterpolation li;

	Double_v x(N), y(N);
	for (int j = 0; j < N; j++)
	{
		x[j] = exp(k_grid_atm[j]);
		y[j] = exp(-alpha * k_grid_atm[j]) / pi * real(x_fft[j]);
		if (!splineFlag) li.AddPoint(x[j], y[j]);

	}

	//spline interpolation
	tk::spline s(x, y);

	Double_v CM_prices(K_grid.size());
	for (int j = 0; j < K_grid.size(); j++)
	{
		double price;

		if (splineFlag) price = s(K_grid[j]);
		else price = li.value(K_grid[j]);

		CM_prices[j] = (CallFlag) ? price : price + K_grid[j] * exp(-r * T) - S0 * exp(-q * T);

		//cout << K_grid[j] << '\t' << price << endl;

	}

	return CM_prices;
}

///////////////////////////FST METHOD///////////////////////////
double LevyMarket::PriceByFST(double K, double T, int N, bool CallFlag, bool splineFlag)
//Double_v FST_European_VarianceGamma(Double_v K_grid, double S0, double r, double T, double sigma, double theta, double nu, double q, int N)
{
	double mu = r + w() - q;

	double FSTprice;
	double price;

	double x_min = -7.5, x_max = 7.5;
	double dx = (x_max - x_min) / (N - 1);
	Double_v x(N);
	for (int j = 0; j < N; ++j) {
		x[j] = x_min + j * dx;
	}

	double w_max = pi / dx;
	double dw = 2 * w_max / N;
	Double_v w(N);
	for (int j = 0; 2 * j <= N; ++j) {
		w[j] = j * dw;
	}
	for (int j = 0; 2 * j < N - 2; ++j) {
		w[int(N / 2) + j + 1] = -w_max + (j + 1) * dw;
	}

	Double_v s(N), v_option(N);
	for (int j = 0; j < N; ++j) {
		s[j] = S0 * exp(x[j]);
		v_option[j] = (CallFlag) ? (max(s[j] - K, 0.0)) : max(K - s[j], 0.0);
	}

	Complex_v fft_v_option(v_option.begin(), v_option.end());

	Complex_v char_exp_factor(N);
	for (int j = 0; j < N; ++j) {
		char_exp_factor[j] = exp((CE(w[j], S0, r, q) - r) * T);
	}

	FFT(fft_v_option);

	for (int j = 0; j < N; ++j) {
		fft_v_option[j] *= char_exp_factor[j];
	}


	IFFT(fft_v_option);

	LinearInterpolation li;
	for (int j = 0; j < N; ++j) {
		v_option[j] = fft_v_option[j].real();
		li.AddPoint(s[j], v_option[j]);
	}

	tk::spline option_spline(s, v_option);

	if (splineFlag) FSTprice = option_spline(S0);
	else FSTprice = li.value(S0);

	return FSTprice;
}

///////////////////////////COS METHOD///////////////////////////
Double_v LevyMarket::PsiCoef(double a, double b, double c, double d, int N)
{
	Double_v psi(N);
	psi[0] = d - c;
	for (int k = 1; k < N; k++)
	{
		double psi_k = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a) / (b - a));
		psi[k] = psi_k * (b - a) / (k * pi);
	}
	return psi;
}

Double_v LevyMarket::ChiCoef(double a, double b, double c, double d, int N)
{
	Double_v chi(N);
	for (int k = 0; k < N; k++)
	{
		chi[k] = 1 / (1 + pow(k * pi / (b - a), 2));

		double expr1 = cos(k * pi * (d - a) / (b - a)) * exp(d)
			- cos(k * pi * (c - a) / (b - a)) * exp(c);

		double expr2 = k * pi / (b - a) * sin(k * pi * (d - a) / (b - a)) * exp(d)
			- k * pi / (b - a) * sin(k * pi * (c - a) / (b - a)) * exp(c);
		chi[k] *= (expr1 + expr2);
	}
	return chi;
}

Double_v LevyMarket::HCoef(double a, double b, int N, bool CallPut)
{
	//Call: CallPut = true 
	//Put:  CallPut = false
	Double_v H(N);

	double c = (CallPut) ? 0 : a;
	double d = (CallPut) ? b : 0;

	Double_v chi = ChiCoef(a, b, c, d, N);
	Double_v psi = PsiCoef(a, b, c, d, N);
	for (int k = 0; k < N; k++)
	{
		double H_k = 2 / (b - a) * (chi[k] - psi[k]) * (CallPut ? 1 : -1);
		H[k] = H_k;
	}
	return H;
}


double LevyMarket::PriceByCOS(double K, double T, int N, double L, bool CallFlag)
{
	double mu = r + w() - q;

	double price = 0;
	double x0 = log(S0 / K);

	//simple range
	double a = -L * sqrt(T);
	double b = L * sqrt(T);

	Double_v H = HCoef(a, b, N, CallFlag);

	for (int k = 0; k < N; k++)
	{
		double u = k * pi / (b - a);

		Complex expr1 = H[k] * CF(u, T, S0, r, q) * exp(-i * u * log(S0));
		Complex expr2 = exp(-i * u * (a - x0));

		if (k == 0) expr1 *= 0.5;
		price += exp(-r * T) * K * real(expr1 * expr2);
	}
	return price;
}



///////////////////////////VG ANALYTICAL FORMULA///////////////////////////
double VarianceGammaMarket::PriceByIntegrationVG(double K, double T, bool CallFlag)
{
	double R_max = 20.0; //integration upper limit
	int M = 20e4;		 //number of steps
	double dR = R_max / M;
	double price = 0;
	for (int k = 0; k < M; k++)
	{
		double R_mid = dR * (0.5 + k);
		double w = 1 / nu * log(1.0 - theta * nu - sigma * sigma * nu / 2);
		double S_tild = S0 * exp(theta * R_mid + w * T + sigma * sigma * R_mid / 2);
		double sigma_tild = sigma * sqrt(R_mid / T);

		double midPrice = BSM_price(K, T, S_tild, r, q, sigma_tild);

		double a = k * dR;
		double b = (k + 1) * dR;

		double gamma_a = boost::math::gamma_p(T / nu, a / nu);
		double gamma_b = boost::math::gamma_p(T / nu, b / nu);
		double gamma = gamma_b - gamma_a;

		price += midPrice * gamma;
	}

	return (CallFlag) ? price : price + K * exp(-r * T) - S0 * exp(-q * T);
}



///////////////////////////CGMY MONTE CARLO///////////////////////////
Double_v CGMYmarket::CGMYdensityByCOS(Double_v x, int N, double T, double a, double b)
{
	Double_v F;
	for (int k = 0; k < N; k++)
	{
		double mu = r + w() - q;
		double u_k = k * pi / (b - a);
		double F_k = 2 / (b - a) * real(CF(u_k, T, S0, r, q) * exp(-i * u_k * (log(S0) + mu * T)) * exp(-i * u_k * a));
		F.push_back(F_k);
	}
	F[0] *= 0.5;

	Double_v f;

	for (auto x_k : x)
	{
		double f_x = 0;
		for (int k = 0; k < N; k++) {
			double u_k = k * pi / (b - a);
			double costerm = cos(u_k * (x_k - a));
			f_x += F[k] * costerm;
		}
		f.push_back(f_x);
	}
	return f;
}

double CGMYmarket::PriceByMonteCarlo(double K, double T, int N_sim, double L, bool CallFlag)
{
	double a = -L * sqrt(T);
	double b = L * sqrt(T);

	int J = int(1e3);
	double x_min = -4;
	double x_max = 4;
	double dx = (x_max - x_min) / J;

	Double_v x(J);
	for (int j = 0; j < J; j++)
	{
		x[j] = x_min + j * dx;
	}

	int N = int(pow(2, 14));

	Double_v f_x = CGMYdensityByCOS(x, N, T, a, b);


	//creating CDF from PDF
	LinearInterpolation li;
	//Double_v F_x;

	//ofstream CGMYdensity("CGMYdensity.txt"); CGMYdensity << "x\tfx\tFx" << endl;
	double F_j = f_x[0];
	for (int j = 0; j < f_x.size(); j++)
	{
		F_j += (j == 0) ? 0 : f_x[j] * dx;
		//CGMYdensity << x[j] << "\t" << f_x[j] << "\t" << F_x[j] << "\n";
		if (0.05 <= F_j && F_j <= 0.95) li.AddPoint(F_j, x[j]);
	}

	double mu = r + w() - q;

	mt19937 RandomEngine(time(NULL));
	uniform_real_distribution <double> UniformRandomNumber(0.0, 1.0);
	double U;

	double price = 0;

	for (int i = 0; i < N_sim; i++)
	{
		U = UniformRandomNumber(RandomEngine);
		double X = li.value(U);
		double S = S0 * exp(mu * T + X);
		double payoff = max(K - S, 0.0);
		price += (payoff * exp(-r * T)) / N_sim;
	}
	return (!CallFlag) ? price : price - K * exp(-r * T) + S0 * exp(-q * T);;
}

