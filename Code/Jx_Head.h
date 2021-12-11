#pragma once
#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include<typeinfo>

using namespace std;

void TensorReshape_permute();

#define test_Jx() TensorReshape_permute()