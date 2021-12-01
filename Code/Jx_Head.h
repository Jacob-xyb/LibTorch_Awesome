#pragma once
#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include<typeinfo>

using namespace std;

void LibNorm_Normalized();

#define test_Jx() LibNorm_Normalized()