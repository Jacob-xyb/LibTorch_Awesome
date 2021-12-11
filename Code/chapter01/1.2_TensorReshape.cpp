#include "../Jx_Head.h"

void TensorReshape_ViewReshape()
{
	// View and reshape neither is in-place, please use 'data = data.view()/data.reshape()'.
	auto x = torch::arange(9);
	cout << "x.view({3,3}):\n" << x.view({ 3,3 }) << endl;
	cout << "x.reshape({3,3}):\n" << x.reshape({ 3,3 }) << endl;

	auto y = x.view({ 3,3 });
	cout << "view_as():\n" << x.view_as(y) << endl;

	// View and reshape both are shallow copy.
	y[0] = torch::tensor({ 8, 8, 8 });
	cout << x.view({ 3,3 }) << endl;

	auto z = x.reshape({ 3,3 });
	z[1] = torch::tensor({ 6, 6, 6 });
	cout << z.view({ 3,3 }) << endl;
}

void TensorReshape_Transpose()
{
	// `.t()` and `.transpose()` both are shallow copy.
	auto x1 = torch::rand({ 2,3 });
	cout << x1 << endl;
	auto x2 = x1.t();
	cout << x2 << endl;
	auto x3 = x1.transpose(0,1);
	cout << x3 << endl;

	TORCH_CHECK(x1.data_ptr() == x2.data_ptr());
	TORCH_CHECK(x1.data_ptr() == x3.data_ptr());

	// But you need to know, after `.t()` and `.transpose()`,
	//	data is not contiguous in storage.
	//	you can't use `view()`.
	//cout << x2.view({ 1,6 }) << endl;
	//cout << x3.view({ 1,6 }) << endl;
	
	// First determine whether it is continuous. `.is_contiguous()`.
	cout << "after .t(), x2.is_contiguous(): " << x2.is_contiguous() << endl;
	x2 = x2.contiguous();
	//	Notice now that the memory space has changed.
	// 	   Means that `.contiguous()` is deep copy.
	//TORCH_CHECK(x1.data_ptr() == x2.data_ptr());
	cout << "after .contiguous(), x2.is_contiguous(): " << x2.is_contiguous() << endl;
	cout << x2.view({ 1,6 }) << endl;
}

void TensorReshape_permute()
{
	auto x = torch::ones({ 1,2,3,4 });
	cout << x.sizes() << endl;
	auto y = x.permute({ 2,3,0,1 });
	cout << y.sizes() << endl;
	TORCH_CHECK(x.data_ptr() == y.data_ptr());
}