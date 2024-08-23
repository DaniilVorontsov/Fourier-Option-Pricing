#pragma once
#include<map>
#include <iostream>

using namespace std;

class LinearInterpolation
{
public:
	void AddPoint(double x, double y);
	double value(double x);

	LinearInterpolation() {};
private:
	map<double, double>DataPoints;
};

