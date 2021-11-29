#include "../Jx_Head.h"


void TensorInitialization_SimpleInitialization()
{
	// Create an empty tensor.
	torch::Tensor x1 = torch::empty({ 2,3 });
	cout << "empty tensor:\n" << x1 << endl;

	// And zeors tensor.
	// We can use 'auto' when we create a tensor.
	auto x2 = torch::zeros({ 2, 3 });
	cout << "zeros tensor:\n" << x2 << endl;
	x2 = torch::ones({ 2,3 });
	cout << "ones tensor:\n" << x2 << endl;

	// And eye tensor.
	//	Square matrix is the default if there is only one parameter.
	//		means: torch::eye(3) == torch::eye(3,3) .
	//	For two arguments, fill in two integers, row and column.
	auto x3 = torch::eye(3);
	cout << "eye(3) tensor:\n" << x3 << endl;
	auto x4 = torch::eye(2, 3);
	cout << "eye(2, 3) tensor:\n" << x4 << endl;
	auto x5 = torch::eye(3, 3);
	cout << "eye(3, 3) tensor:\n" << x5 << endl;

	// And constant tensor.
	auto x6 = torch::full({ 2,3 }, 6.66);
	cout << "constant tensor:\n" << x6 << endl;
}

void TensorInitialization_RandomInitialization()
{
	// Rand() produces a random value between 0 and 1.
	auto x1 = torch::rand({ 2,3 });
	cout << "rand({2,3}):\n" << x1 << endl;

	// Randint(high) produces a random int in [0,high).
	auto x2 = torch::randint(2, { 2,3 });
	cout << "randint(2):\n" << x2 << endl;
	// Randint(low,high) produces a random int in [low,high).
	auto x3 = torch::randint(-2, 2, { 2, 3 });
	cout << "randint(-2, 2, { 2, 3 }):\n" << x3 << endl;

	// Randn() takes the random value of the normal distribution N(0,1).
	auto x4 = torch::randn({ 2,3 });
	cout << "randn tensor:\n" << x4 << endl;
}

void TensorInitialization_LineInitialization()
{
	// Normal operation: linspace(start, end, nums);
	//	you will get nums values line in [start, end].
	auto x1 = torch::linspace(1, 5, 10);
	cout << x1.view({ 1,-1 }) << endl;

	// If you only input 2 params, you will get the following warnings, but the program will finish.
	//	Like 'auto x1 = torch::linspace(1,5);` .
	/*Warning: Not providing a value for linspace's steps is deprecated and will throw a runtime error in a future release. This warning will appear only once per process. (function operator ())*/

	//**********************************************//

	// Arange is an ordered sequence of integers and there are three common ways to call it.
	//	Very like Python's Numpy.
	//		But first you should know: range of sequence is [start, end).
	// 1.arange(strat, end, steps); return LongTensor [start:end-1:steps]. 
	auto x2 = torch::arange(0, 6, 2);
	cout << x2.view({ 1,-1 }) << endl;
	// 2.arange(strat, end); return LongTensor [start:end-1].
	auto x3 = torch::arange(3, 6);
	cout << x3.view({ 1,-1 }) << endl;
	// 3.arange(Scalar); return LongTensor [0:Scalar-1].
	auto x4 = torch::arange(5);
	cout << x4.view({ 1,-1 }) << endl;
}

void TensorInitialization_OtherData()
{
	//Tips:
	//	torch::tensor() is deep copy; torch::from_blob is shallow copy. 

	//array
	//	I don't think arrays are very well supported,
	//	so it's good to use vectors.
	//	Because array is a point, you can think of it like a Long int, so array is not good.
	int a[3] = { 1,2,3 };
	int b[3] = { 0,2,1 };
	auto x1 = torch::tensor(a);
	auto x2 = torch::tensor(b);
	cout << x1 << endl << x2 << endl;
	//	Even if you use 'blob',you must know the type of point.
	//	Like this:
	//		If you transform to other dtype, you will not get true data.
	auto x3 = torch::from_blob(a, { 3 }, torch::kLong);
	cout << x3 << endl;
	//		Only true dtype can be true data, it's painful.
	auto x4 = torch::from_blob(a, { 3 }, torch::kInt);
	cout << x4 << endl;

	cout << endl << "*************************" << endl;

	//vector
	//	vector is very easy.
	vector<int> va = { 3, 2, 1 };
	auto x5 = torch::tensor(va);
	cout << x5 << endl;
	//	Tips;because blob's input is point, so must input 'size' param.
	auto x6 = torch::from_blob(&va[0], { 1,3 }, torch::kInt);
	//		Same as:
	//auto x6 = torch::from_blob(va.data(), { 1,3 }, torch::kInt);
	cout << x6 << endl;
}

void TensorInitialization_Like()
{
	// There are many similar functions as '_like()'.
	auto x = torch::rand({ 2,3 });
	auto x1 = torch::rand_like(x);
	cout << "rand_like():\n" << x1 << endl;
	auto x2 = torch::randint_like(x, 2);
	cout << "randint_like():\n" << x2 << endl;
	auto x3 = torch::zeros_like(x);
	cout << "zeros_like():\n" << x3 << endl;

	cout << endl << "*************************" << endl;

	// But the way of torch::Tensor(), note that this is the constructor of the class!
	//	This way is shallow copy, yes, shallow copy, uhhh...
	auto y = torch::ones({ 2,3 });
	auto y1 = torch::Tensor(y);
	y1[0] = 6;
	cout << y << endl;
	// Maybe you can use clone() avoid this situation.
	auto y2 = torch::Tensor(y).clone();
	y2[0] = 8;
	cout << y << endl;
}
