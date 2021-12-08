#include "../Jx_Head.h"

void BasicCaculation()
{
	vector<int> idx(5);
	for (int i = 0; i < 5; i++)
	{
		idx[i] = i;
	}
	random_shuffle(idx.begin(), idx.end());
	auto index = torch::tensor(idx);
	cout << index << endl;
	auto x = torch::arange(100).view({ 10 ,-1 });
	//cout << x.index_select(0, idx) << endl;
}

void BasicCaculation_Arithmetic()
{
	// +-*/ all are same. If same of size, caculation of corresponding elements.
	auto x = torch::randint(10, { 2,4 });
	cout << x << endl;
	cout << x + 1 << endl;

	auto y = torch::randint_like(x, 10);
	cout << y << endl;
	cout << x + y << endl;

	// Basic broadcasting.
	auto z = torch::tensor({ 1,2,3,4 });
	cout << x + z << endl;
}

void BasicCaculation_Matrix()
{
	// square
	auto m1 = torch::full({ 3,3 }, 2.);
	auto m2 = m1.clone();
	// Multiplication and matrix multiplication. `*` == `.mul()`.
	cout << m1 * m2 << endl;
	cout << m1.mul(m2) << endl;
	cout << m1.matmul(m2) << endl;
}