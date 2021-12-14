#pragma once
#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include<typeinfo>

using namespace std;

void SlicingOperation_gather();

#define test_Jx() SlicingOperation_gather()