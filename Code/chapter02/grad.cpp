#include "../Jx_Head.h"

void LibGrad_Setting()
{
	auto x = torch::tensor({ 1,2,3 }, torch::kFloat);
	// x.requires_grad();		// If you use this way, grad is not in-place.
	x.requires_grad_();
	cout << x.grad_fn() << endl;		// x is a leaf node, so not grad_fn.

	auto y = x + 2;
	cout << y.grad_fn() << endl;		// Now, grad_fn is effect.

	// Check whether it is a leaf node.
	cout << x.is_leaf() << "\t" << y.is_leaf() << endl;
}

void LibGrad_Backward()
{
	// Grad is accumulated in back propagation.
	//	Tips;must be Scalar.backward().
	auto x = torch::ones({ 2,2 });
	cout << x << endl;
}