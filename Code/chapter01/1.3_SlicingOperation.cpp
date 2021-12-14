#include "../Jx_Head.h"

void SlicingOperation()
{
	// There is one-to-one correspondence between Python and C++ tensor index types:
	// Python                  | C++
	// -----------------------------------------------------
	// `None`                  | `at::indexing::None`
	// `Ellipsis`              | `at::indexing::Ellipsis`
	// `...`                   | `"..."`
	// `123`                   | `123`
	// `True` / `False`        | `true` / `false`
	// `:`                     | `Slice()` / `Slice(None, None)`
	// `::`                    | `Slice()` / `Slice(None, None, None)`
	// `1:`                    | `Slice(1, None)`
	// `1::`                   | `Slice(1, None, None)`
	// `:3`                    | `Slice(None, 3)`
	// `:3:`                   | `Slice(None, 3, None)`
	// `::2`                   | `Slice(None, None, 2)`
	// `1:3`                   | `Slice(1, 3)`
	// `1::2`                  | `Slice(1, None, 2)`
	// `:3:2`                  | `Slice(None, 3, 2)`
	// `1:3:2`                 | `Slice(1, 3, 2)`
	// `torch.tensor([1, 2])`) | `torch::tensor({1, 2})`

	//x.slice(0, c10::nullopt);
}

void SlicingOperation_Basic()
{
	auto x = torch::rand({ 3, 3, 5 });
	cout << x << endl;
	
	// Index data like std::vector use operator [].
	cout << "channel 0:x[0]\n" << x[0] << endl;
	cout << "channel 1:x[0][0]\n" << x[0][0] << endl;
	cout << "channel 2:x[0][0][0]\n" << x[0][0][0] << endl;
}


void SlicingOperation_Slice()
{
	// slice(dim,start,end,step);shallow copy.
	auto x = torch::rand({ 3, 3, 5 });
	cout << x << endl;
	cout << x.slice(1, 1) << endl;
	
	//auto y = x.slice(1, 1);
	//y[0] = 3;
	//cout << x << endl;
}

void SlicingOperation_Index()
{
	// There are many methods to call.
	auto x = torch::rand({ 3, 3, 5 });
	cout << x << endl;

	// Method 1: x.index({int, int, ...})
	//	This way is suitable to fetch only one dimension of data.
	cout << x.index({ 0, 1, 1 }) << endl;
	vector<int> v{ 1,2 };
	//		But x.index(vector) or x.index({vector}) both are error.

	cout << endl << "*************************" << endl;

	// Method 2: x.index({tensor}), but has many ways to call.
	//	Way 1: x.index({tensor})
	//		This way returns a sequence in dim 0.
	auto idx1 = torch::tensor({ 0,1 });
	cout << "x.index({tensor}):\n" << x.index({idx1}) << endl;
	//		But x.index(tensor) is error.

	//	Way 2: x.index({tensor, int})
	//		This way returns a sequence tensor in dim 0, and int in dim 1.
	cout << "x.index({tensor, int}):\n" << x.index({idx1, 1}) << endl;

	//	Way 3: x.index({tensor1,tensor2})
	//		This way returns a sequence [tensor1[0] in dim 0, tensor2[0] in dim 1],
	//							a sequence [tensor1[1] in dim 0, tensor2[1] in dim 1]...
	cout << "x.index({tensor1,tensor2}):\n" << x.index({idx1, idx1}) << endl;
	auto idx2 = torch::tensor({ 0,1,2 });
	//		So tensor1.sizes() must same as tensor2.sizes(), otherwise error.
	//cout << x.index({ idx1, idx2 }) << endl;		//error

	// Way 4: x.index({"...", int})
	//		This way returns a sequence all but int in dim last.Just last.
	//		And you will find that cols -> rows.
	cout << "x.index({\"...\", int}):\n" << x.index({ "...", 1 }) << endl;
}

void SlicingOperation_Select()
{
	// select(dim,index)
	// Select is very easy, not suitable for dim >= 2;
	//	If dim >=2 , returns every index in dim.
	//	If you want to copy, add '.clone()'.
	auto x = torch::rand({ 3, 3, 5 });
	cout << x << endl;
	cout << x.select(1, 1) << endl;
	auto x1 = torch::select(x, 1, 2);
	cout << x1 << endl;
}

void SlicingOperation_IndexSelect()
{
	auto x = torch::rand({ 3, 3, 5 });
	cout << x << endl;
	auto idx = torch::tensor({ 0,1 });
	cout << x.index_select(1, idx) << endl;
}

void SlicingOperation_IndexExpand()
{
	auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
	auto x = torch::tensor({ {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} }, options);
	using torch::indexing::Slice;
	auto _None = torch::indexing::None;
	auto _Ellipsis = torch::indexing::Ellipsis;  
	cout << " x[0,2] = " << x.index({ 0, 2 }) << endl;
	// x[,2] has 3 ways to display.
	cout << " x[,2] = \n" << x.index({ Slice(), 2 }).unsqueeze(0) << endl;
	cout << " x[,2] = \n" << x.index({ Slice(_None), 2 }).unsqueeze(0) << endl;		// Slice() == Slice(_None) == _Ellipsis
	cout << " x[,2] = \n" << x.index({ _Ellipsis, 2 }).unsqueeze(0) << endl;

	cout << " x[1,:2] = \n" << x.index({ 1, Slice(_None, 2) }).unsqueeze(0) << endl;
	cout << "\"x[:,::2]\":\n" << x.index({ Slice(), Slice(_None, _None, 2) }) << endl;
}

void SlicingOperation_chunk()
{
	// torch::chunk(tensor, chunks, dim=0) ¡ú List of Tensors
	// tensor.chunk(tensor, chunks, dim=0) ¡ú List of Tensors

	// Shallow copy. IF want to change value, please .clone().
	// Splits a tensor into a specific number of chunks.
	// Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.
	auto x = torch::randn({ 4, 4 });
	cout << x << endl;
	auto y1 = x.chunk(2);
	cout << y1 << endl;
	auto y2 = x.chunk(2, 1);
	cout << y2 << endl;
	cout << y2[0] << endl;
	TORCH_CHECK(x.data_ptr() == y1[0].data_ptr());
}

void SlicingOperation_gather()
{
	// at::Tensor at::gather(const at::Tensor &self, int64_t dim, const at::Tensor &index, bool sparse_grad = false)

	// The torch::gather() function is not easy to understand from a formula.
	// Just for a quick demonstration, index() is good.
	auto x = torch::tensor({ {1,2,3}, {4,5,6} });
	auto idx = torch::tensor({ { 2. },{ 1. } }).to(torch::kLong);
	cout << x.gather(1, idx) << endl;
}


