//  Squeezenet.cpp
//  MNN
//
//  Created by CDQ on 2021/03/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "Squeezenet.hpp"
#include <MNN/expr/NN.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

Squeezenet::Squeezenet(){
    NN::ConvOption convOption;
    convOption.kernelSize = {7, 7};
    convOption.channel    = {1, 96};
    convOption.stride     = {2, 2};
    convOption.padMode    = Express::SAME;
    conv1.reset(NN::Conv(convOption));
    bn1.reset(NN::BatchNorm(96));

    // 第一次写的时候这里都是conv1.reset，导致registerModel的时候失败
    convOption.kernelSize = {1, 1};
    convOption.channel    = {512, 10};
    convOption.padMode    = Express::VALID;
    conv2.reset(NN::Conv(convOption));
    bn2.reset(NN::BatchNorm(10));

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


    fire1 = FireMoudle(96, 16, 64, 64);
    fire2 = FireMoudle(128, 16, 64, 64);
    fire3 = FireMoudle(128, 32, 128, 128);
    fire4 = FireMoudle(256, 32, 128, 128);
    fire5 = FireMoudle(256, 48, 192, 192);
    fire6 = FireMoudle(384, 48, 192, 192);
    fire7 = FireMoudle(384, 64, 256, 256);
    fire8 = FireMoudle(512, 64, 256, 256);

    registerModel({conv1, conv2, conv_mp1, conv_mp2, conv_mp3, conv_ap,
                   fire1, fire2, fire3, fire4, fire5, fire6, fire7, fire8, bn1, bn2});
    // registerModel({conv1, conv2, conv3});
    setName("Squeezenet");
}

std::vector<Express::VARP> Squeezenet::onForward(const std::vector<Express::VARP>& inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x      = conv1->forward(x);
    x      = bn1->forward(x);
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
    x      = bn2->forward(x);
    x      = conv_ap->forward(x);
    x      = _Convert(x, NCHW);
    x      = _Reshape(x, {0, -1});
    x      = _Softmax(x);
    return {x};
}

// // fire module in Squeezenet model
// static VARP fireMoudle(VARP x, int inputChannel, int squeeze_1x1,
//                        int expand_1x1, int expand_3x3) {
//     x = _Conv(0.0f, 0.0f, x, {inputChannel, squeeze_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     auto y1 = _Conv(0.0f, 0.0f, x, {squeeze_1x1, expand_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     auto y2 = _Conv(0.0f, 0.0f, x, {squeeze_1x1, expand_3x3}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
//     return _Concat({y1, y2}, 1); // concat on channel axis (NCHW)
// }

// VARP SqueezenetExpr(int numClass) {
//     auto x = _Input({1, 3, 224, 224}, NC4HW4);
//     x = _Conv(0.0f, 0.0f, x, {3, 96}, {7, 7}, SAME, {2, 2}, {1, 1}, 1);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = fireMoudle(x, 96, 16, 64, 64);
//     x = fireMoudle(x, 128, 16, 64, 64);
//     x = fireMoudle(x, 128, 32, 128, 128);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = fireMoudle(x, 256, 32, 128, 128);
//     x = fireMoudle(x, 256, 48, 192, 192);
//     x = fireMoudle(x, 384, 48, 192, 192);
//     x = fireMoudle(x, 384, 64, 256, 256);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = fireMoudle(x, 512, 64, 256, 256);
//     x = _Conv(0.0f, 0.0f, x, {512, numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     x = _AvePool(x, {14, 14}, {1, 1}, VALID);
//     return x;
// }


} // namespace Model
} // namespace Train
} // namespace MNN
