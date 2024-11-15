**Fourier transform methods for option pricing**  
The MSc thesis was concerned with option pricing under exponential Lévy jump models. Three renowned methods were implemented in C++ for Variance Gamma and CGMY processes. For CGMY additional study was conducted to evaluate option prices through Monte-Carlo simulation accelerated on GPU with CUDA Nvidia. For further information please refer to the PDF report.

**The contents of the project**  
One base class LevyMarket contains all three Fourier transform pricing methods implementations. LevyMarket contains virtual functions CF (Characteristic Function) and CE (Characteristic Exponent) which are defined in inheritance classes VarianceGammaMarket and CGMYmarket. VarianceGammaMarket class corresponds to the Variance Gamma model. Apart from CF and CE, it has a method for analytical price integration. CGMYmarket class corresponds to the CGMY model. Apart from CF and CE, it has a CPU implementation of MCFT-1.
CUDA part is located in .cu files where functions MCFT1 and MCFT2 are defined. This functions are called in files VG.cpp and CGMY.cpp. For VG analytical pricing Boost library is required.

main.cpp
-- main function

VG.cpp
-- error tests and time performance for two sets of options under VG

CGMY.cpp
-- error tests and time performance for two sets of options under CGMY

FFT.h, FFT.cpp 
-- implementation of a Fast Fourier Transform (radix-2 DIT Cooley–Tukey algorithm)

Spline.h
-- interploation library by Tino Kluge https://kluge.in-chemnitz.de/opensource/spline/

LinearInterpolation.h, LinearInterpolation.cpp
-- linear interpolation class

FourierMethods.h, FourierMethods.cpp
-- Classes for Fourier methods, auxilary functions for MCFT1, MCFT2
-- Fourier methods

MCFT1.cu
-- MCFT1 implementation with CUDA, requires CUDA 

MCFT2.cu
-- MCFT2 implementation with CUDA, requires CUDA 

Visualization.ipynb
-- plots and aggregation for report

VG_CGMY.ipynb
-- CDF recovery for CGMY and sampling of GBM, VG, CGMY plots for report

Release version x64
ISO C++14 Standard
Intel(R) Core(TM) i5-11400H 2.70 GHz, 12 CPU
RTX 3050 Ti Laptop GPU
CUDA version: release 11.8, V11.8.89
Boost Version 1.85.0
Microsoft Visual C++ (MSVC)
