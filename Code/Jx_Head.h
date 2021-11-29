#pragma once
#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include<typeinfo>

using namespace std;

void TensorInitialization_Like();

#define test_Jx() TensorInitialization_Like()