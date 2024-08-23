#include "LinearInterpolation.h"


void LinearInterpolation::AddPoint(double x, double y) {
	DataPoints.insert_or_assign(x, y);
}

double LinearInterpolation::value(double x)
{
	//in case x value presented within DataPoints:
	if (DataPoints.find(x) != DataPoints.end())
	{
		return DataPoints.find(x)->second;
	}

	if (DataPoints.size() == 0 || DataPoints.size() == 1)
	{
		cout << "ERROR! No values were passed or only one";
		exit(-100);
	}

	//in case of out of boundaries assign closest value

	if (x < DataPoints.begin()->first) return DataPoints.begin()->second;

	auto it_end = DataPoints.end();
	it_end--;
	
	if (x > it_end->first) return it_end->second;

	for (map<double, double>::iterator it = DataPoints.begin(); it != DataPoints.end(); ++it)
	{
		if (x < it->first)
		{
			double T = it->first;
			double y_T = it->second;
			it--;
			double t = it->first;
			double y_t = it->second;
			return (y_T - y_t) / (T - t) * (x - t) + y_t;
		}
	}

}