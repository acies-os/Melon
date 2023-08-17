//  SqueezenetNoBN.cpp
//  MNN
//
//  Created by CDQ on 2021/03/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "SqueezenetNoBN.hpp"
#include <MNN/expr/NN.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

SqueezenetNoBN::SqueezenetNoBN(){
    NN::ConvOption convOption;
    convOption.kernelSize = {7, 7};
    convOption.channel    = {1, 96};
    convOption.stride     = {2, 2};
    convOption.padMode    = Express::SAME;
    conv1.reset(NN::Conv(convOption));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {512, 10};
    convOption.padMode    = Express::VALID;
    conv2.reset(NN::Conv(convOption));

    convOption.kernelSize = {3, 3};
    convOption.stride     = {2, 2};
    convOption.padMode    = Express::SAME;
    convOption.depthwise  = true;

    convOption.channel    = {96, 96};
    conv_mp1.reset(NN::Conv(convOption));

    convOption.channel    = {256, 256};
    conv_mp2.reset(NN::Conv(convOption));

    convOption.channel    = {512, 512};
    conv_mp3.reset(NN::Conv(convOption));

    convOption.kernelSize = {7, 7};
    convOption.stride     = {7, 7};
    convOption.channel    = {10, 10};
    conv_ap.reset(NN::Conv(convOption));


    fire1 = FireMoudleNoBN(96, 16, 64, 64);
    fire2 = FireMoudleNoBN(128, 16, 64, 64);
    fire3 = FireMoudleNoBN(128, 32, 128, 128);
    fire4 = FireMoudleNoBN(256, 32, 128, 128);
    fire5 = FireMoudleNoBN(256, 48, 192, 192);
    fire6 = FireMoudleNoBN(384, 48, 192, 192);
    fire7 = FireMoudleNoBN(384, 64, 256, 256);
    fire8 = FireMoudleNoBN(512, 64, 256, 256);

    registerModel({conv1, conv2, conv_mp1, conv_mp2, conv_mp3, conv_ap,
                   fire1, fire2, fire3, fire4, fire5, fire6, fire7, fire8});
    // registerModel({conv1, conv2, conv3});
    setName("SqueezenetNoBN");
}

std::vector<Express::VARP> SqueezenetNoBN::onForward(const std::vector<Express::VARP>& inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x      = conv1->forward(x);
    x      = conv_mp1->forward(x);
    x      = fire1->forward(x);
    x      = fire2->forward(x);
    x      = fire3->forward(x);
    x      = conv_mp2->forward(x);
    x      = fire4->forward(x);
    x      = fire5->forward(x);
    x      = fire6->forward(x);
    x      = fire7->forward(x);
    x      = conv_mp3->forward(x);
    x      = fire8->forward(x);
    x      = conv2->forward(x);
    x      = conv_ap->forward(x);
    x      = _Convert(x, NCHW);
    x      = _Reshape(x, {0, -1});
    x      = _Softmax(x);
    return {x};
}


} // namespace Model
} // namespace Train
} // namespace MNN
