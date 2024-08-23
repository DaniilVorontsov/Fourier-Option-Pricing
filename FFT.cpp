#include "FFT.h"

void FFT(Complex_v& f) {
	int n = f.size();
	if (n <= 1) return;

	Complex_v even(n / 2);
	Complex_v odd(n / 2);

	for (int i = 0; i < n / 2; ++i)
	{
		even[i] = f[i * 2];
		odd[i] = f[i * 2 + 1];
	}

	FFT(even);
	FFT(odd);

	for (int i = 0; 2 * i < n; ++i) {
		complex<double> t = polar(1.0, -2 * pi * i / n) * odd[i];
		f[i] = even[i] + t;
		f[i + n / 2] = even[i] - t;
	}
}


void IFFT(Complex_v& F) {
	int n = F.size();
	if (n <= 1) return;

	for (auto& x : F) x = conj(x);
	FFT(F);
	for (auto& x : F) x = conj(x);

	for (auto& x : F) x /= n;
}