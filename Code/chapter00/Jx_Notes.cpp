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