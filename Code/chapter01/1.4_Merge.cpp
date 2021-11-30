#include "../Jx_Head.h"

void LibMerge()
{

}

void LibMerge_Cat()
{
	//torch::cat(TensorList, dim)
	auto x = torch::randint(10, { 3,5 });
	auto y = torch::randint(-10, 0, { 3,5 });
	cout << torch::cat({ x,y }, 0) << endl;
	cout << torch::cat({ x,y }, 1) << endl;
}

void LibMerge_CatVector()
{
	auto x = torch::randint(10, { 2,3 });
	vector<torch::Tensor> y = { x, x, x };
	cout << torch::cat(y, 0) << endl;
}