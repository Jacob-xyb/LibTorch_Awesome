#include "../Jx_Head.h"

//typeid
void HowToType()
{
	// typeid() is not very useful for LibTorch
	auto x = torch::rand({ 2,3 });
	auto y = torch::tensor(2);
	cout << y.sizes() << endl;
	cout << typeid(x).name() << endl;
	cout << typeid(y).name() << endl;
}

void HowToPtr()
{
	auto x = torch::rand({ 2,3 });
	cout << x.data_ptr<float>() << endl;
}

void Jx_TODO()
{
	// torch::linear return scalar ; like dot product.
	auto x = torch::tensor({ 1,2,3 });
	auto w = torch::tensor({ 2,2,2 });
	auto x1 = torch::linear(x, w);
	cout << x1 << endl;
}