#include "../Jx_Head.h"

void TensorReshape_ViewReshape()
{
	// View and reshape neither is in-place, please use 'data = data.view()/data.reshape()'.
	auto x = torch::arange(9);
	cout << "x.view({3,3}):\n" << x.view({ 3,3 }) << endl;
	cout << "x.reshape({3,3}):\n" << x.reshape({3,3}) << endl;

	auto y = x.view({ 3,3 });
	cout << "view_as():\n" << x.view_as(y) << endl;
}