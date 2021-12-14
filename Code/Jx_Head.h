#pragma once
#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include<typeinfo>

using namespace std;

void SlicingOperation_chunk();

#define test_Jx() SlicingOperation_chunk()