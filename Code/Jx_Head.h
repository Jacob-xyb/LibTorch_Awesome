#pragma once
#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include<typeinfo>

using namespace std;

void TensorInfo_DtypeInfo();

#define test_Jx() TensorInfo_DtypeInfo()