#include "../Jx_Head.h"

void LibNorm_MostValue()
{
	auto x = torch::randint(10, { 3,5 });
	cout << x << endl;
	cout << x.max() << endl;
	cout << x.amax(0) << endl;
	cout << x.amin(0) << endl;
	cout << std::get<0>(x.max(0)) << endl;		// return maxs values
	cout << std::get<1>(x.max(0)) << endl;		// return maxs index
}

void LibNorm_Normalized()
{
	auto x = torch::randint(10, { 3,5 });
	auto x_max = x.amax(0);
	auto x_min = x.amin(0);
	auto x_mean = (x_max - x_min) / 2.;
	auto x_std = x_max - x_mean;
	cout << x << endl;
	cout << x_std << endl;
	cout << (x - x_mean)/ x_std << endl;
}