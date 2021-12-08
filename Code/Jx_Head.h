#pragma once
#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include<typeinfo>

using namespace std;

void LibNorm_MostValue();

#define test_Jx() LibNorm_MostValue()