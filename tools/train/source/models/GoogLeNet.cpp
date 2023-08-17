//  GoogLeNet.cpp
//  MNN
//
//  Created by CDQ on 2021/03/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "GoogLeNet.hpp"
#include <MNN/expr/NN.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

GoogLenet::GoogLenet(){
    NN::ConvOption convOption;
    convOption.kernelSize = {7, 7};
    convOption.channel    = {1, 64};
    convOption.stride     = {2, 2};
    convOption.padMode    = Express::SAME;
    conv1.reset(NN::Conv(convOption));
    bn1.reset(NN::BatchNorm(64));
    
    // 第一次写的时候这里都是conv1.reset，导致registerModel的时候失败
    convOption.kernelSize = {3, 3};
    convOption.channel    = {64, 192};
    convOption.padMode    = Express::SAME;
    conv2.reset(NN::Conv(convOption));
    bn2.reset(NN::BatchNorm(192));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {1024, 10};
    convOption.padMode    = Express::VALID;
    conv3.reset(NN::Conv(convOption));
    bn3.reset(NN::BatchNorm(10));

    convOption.kernelSize = {3, 3};
    convOption.stride     = {2, 2};
    convOption.padMode    = Express::SAME;
    convOption.depthwise  = true;

    convOption.channel    = {64, 64};
    conv_mp1.reset(NN::Conv(convOption));

    convOption.channel    = {192, 192};
    conv_mp2.reset(NN::Conv(convOption));

    convOption.channel    = {480, 480};
    conv_mp3.reset(NN::Conv(convOption));

    convOption.channel    = {832, 832};
    conv_mp4.reset(NN::Conv(convOption));

    convOption.kernelSize = {4, 4};
    convOption.stride     = {4, 4};
    convOption.padMode    = Express::SAME;
    convOption.depthwise  = true;
    convOption.channel    = {1024, 1024};
    conv_ap.reset(NN::Conv(convOption));

    incep1 = Inception(192, 64, 96, 128, 16, 32, 32);
    incep2 = Inception(256, 128, 128, 192, 32, 96, 64);
    incep3 = Inception(480, 192, 96, 208, 16, 48, 64);
    incep4 = Inception(512, 160, 112, 224, 24, 64, 64);
    incep5 = Inception(512, 128, 128, 256, 24, 64, 64);
    incep6 = Inception(512, 112, 144, 288, 32, 64, 64);
    incep7 = Inception(528, 256, 160, 320, 32, 128, 128);
    incep8 = Inception(832, 256, 160, 320, 32, 128, 128);
    incep9 = Inception(832, 384, 192, 384, 48, 128, 128);

    
    registerModel({conv1, conv2, conv3, incep1, incep2, incep3, incep4, incep5, incep6,
                   incep7, incep8, incep9, conv_mp1, conv_mp2, conv_mp3, conv_mp4, conv_ap, bn1, bn2, bn3});
    // registerModel({conv1, conv2, conv3});
    setName("Googlenet");
}

std::vector<Express::VARP> GoogLenet::onForward(const std::vector<Express::VARP>& inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x      = conv1->forward(x);
    x      = bn1->forward(x);
    x      = conv_mp1->forward(x);
    x      = conv2->forward(x);
    x      = bn2->forward(x);
    x      = conv_mp2->forward(x);
    x      = incep1->forward(x);
    x      = incep2->forward(x);
    x      = conv_mp3->forward(x);
    x      = incep3->forward(x);
    x      = incep4->forward(x);
    x      = incep5->forward(x);
    x      = incep6->forward(x);
    x      = incep7->forward(x);
    x      = conv_mp4->forward(x);
    x      = incep8->forward(x);
    x      = incep9->forward(x);
    x      = conv_ap->forward(x);
    x      = conv3->forward(x);
    x      = bn3->forward(x);
    x      = _Softmax(x, -1);
    x      = _Reshape(x, {0, -1});
    return {x};
}

// Express::VARP GoogLenet::inception(VARP x, int inputChannelSet, int channel_1x1,
//                       int channel_3x3_reduce, int channel_3x3,
//                       int channel_5x5_reduce, int channel_5x5,
//                       int channel_pool) {
//     auto inputChannel = x->getInfo()->dim[1];
//     auto y1 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     auto y2 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_3x3_reduce}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     y2 = _Conv(0.0f, 0.0f, y2, {channel_3x3_reduce, channel_3x3}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
//     auto y3 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_5x5_reduce}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     y3 = _Conv(0.0f, 0.0f, y3, {channel_5x5_reduce, channel_5x5}, {5, 5}, SAME, {1, 1}, {1, 1}, 1);
//     auto y4 = _MaxPool(x, {3, 3}, {1, 1}, SAME);
//     y4 = _Conv(0.0f, 0.0f, y4, {inputChannel, channel_pool}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     return _Concat({y1, y2, y3, y4}, 1); // concat on channel axis (NCHW)
// }

// std::vector<Express::VARP> GoogLenet::onForward(const std::vector<Express::VARP>& inputs) {
//     using namespace Express;
//     auto x = inputs[0];
//     x = _Conv(0.0f, 0.0f, x, {1, 64}, {7, 7}, SAME, {2, 2}, {1, 1}, 1);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = _Conv(0.0f, 0.0f, x, {64, 192}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = inception(x, 192, 64, 96, 128, 16, 32, 32);
//     x = inception(x, 256, 128, 128, 192, 32, 96, 64);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = inception(x, 480, 192, 96, 208, 16, 48, 64);
//     x = inception(x, 512, 160, 112, 224, 24, 64, 64);
//     x = inception(x, 512, 128, 128, 256, 24, 64, 64);
//     x = inception(x, 512, 112, 144, 288, 32, 64, 64);
//     x = inception(x, 512, 256, 160, 320, 32, 128, 128);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = inception(x, 832, 256, 160, 320, 32, 128, 128);
//     x = inception(x, 832, 384, 192, 384, 48, 128, 128);
//     x = _AvePool(x, {7, 7}, {1, 1}, VALID);
//     x = _Conv(0.0f, 0.0f, x, {1024, 10}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // replace FC with Conv1x1
//     x = _Softmax(x, -1);
//     return {x};
// }

} // namespace Model
} // namespace Train
} // namespace MNN
