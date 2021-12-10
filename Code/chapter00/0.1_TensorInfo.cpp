#include "../Jx_Head.h"

void TensorInfo()
{
	torch::Tensor x;
	cout << x << endl;
	if (x.numel())
	{
		cout << "xxx" << endl;
	}
}

void TensorInfo_BasicInfo()
{
	auto x = torch::rand({ 2,3 });
	cout << x << endl;
	cout << x.storage().data() << endl;		// get address by `.data()`
	cout << x.data_ptr<float>() << endl;	// `.storage().data()` == `.data_ptr<Type>()`
	cout << x.data_ptr() << endl;			// you can use implic to cout only when cout, but not recommend.
	float* p1 = x.data_ptr<float>();
	cout << *(p1 + 1) << endl;

	// storage_offset and strides
	auto second_point = x[0][1];
	cout << "second_point.sizes() = " << second_point.sizes() << "\n";
	cout << "second_point.storage_offset() = " << second_point.storage_offset() << "\n";
	cout << "x.strides() = " << x.strides() << "\n";
}

void TensorInfo_DtypeInfo()
{
	auto x = torch::rand({ 2,3 });
	cout << x.dtype() << endl;				// `.dtype()` return `caffe2::TypeMeta`
	cout << x.dtype().name() << endl;		// `.dtype().name()` return `c10::string_view`
	string t = string(x.dtype().name());	// only this can be converted to string.
	cout << t << endl;
}