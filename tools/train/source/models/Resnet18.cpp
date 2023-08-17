//
//  .cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef Resnet18_cpp
#define Resnet18_cpp

#include "Resnet18.hpp"
#include "Initializer.hpp"
#include <MNN/expr/NN.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {

Resnet18::Resnet18(int numClass) {
}

std::vector<Express::VARP> Resnet18::onForward(const std::vector<Express::VARP>& inputs) {
    return {};
}


} // namespace Model
} // namespace Train
} // namespace MNN
#endif
