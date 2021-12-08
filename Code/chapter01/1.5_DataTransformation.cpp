#include "../Jx_Head.h"

void DataTransformation()
{

}

void DataTransformation_Vector()
{
	// vector to tensor is very easy.
	vector<float> v1 = { 1, 2, 3 };
	torch::Tensor x1 = torch::tensor(v1);

	// tensor to vector maybe be problems
	vector<float> v2(x1.data_ptr<float>(), x1.data_ptr<float>() + x1.numel());
	//	Why not transform by self?
	vector<float> v3;
	x1.unsqueeze(0);
	for (int i = 0; i < x1.numel(); i++)
	{
		v3.emplace_back(x1[i].item<float>());
	}
}