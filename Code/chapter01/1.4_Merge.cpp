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

void LibMerge_stack()
{
	// at::Tensor at::stack(at::TensorList tensors, int64_t dim = 0)
	// Join the sequence of input tensors along a new dimension. All tensors in a sequence should have the same shape.
	auto x1 = torch::randint(10, { 2,2 });
	auto x2 = torch::randint(10, { 2,2 });
	auto x3 = torch::randint(10, { 2,2 });
	auto x = torch::stack({ x1,x2,x3 }, 0);
	cout << x1 << "\n" << x2 << "\n" << x3 << endl;
	cout << x << endl;
	x = torch::stack({ x1,x2,x3 }, 1);
	cout << x << endl;
	x = torch::stack({ x1,x2,x3 }, 2);
	cout << x << endl;
}