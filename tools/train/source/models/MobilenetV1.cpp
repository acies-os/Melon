//
//  MobilenetV1.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MobilenetV1_cpp
#define MobilenetV1_cpp

#include "MobilenetV1.hpp"
#include "Initializer.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {

MobilenetV1::MobilenetV1(int numClasses, float widthMult, int divisor) {
    NN::ConvOption convOption;
    convOption.kernelSize = {3, 3};
    int outputChannels    = makeDivisible(32 * widthMult, divisor);
    convOption.channel    = {3, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {2, 2};
    conv1.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    bn1.reset(NN::BatchNorm(outputChannels));

    std::vector<std::vector<int> > convSettings;
    convSettings.push_back({64, 1});
    convSettings.push_back({128, 2});
    convSettings.push_back({256, 2});
    convSettings.push_back({512, 6});
    convSettings.push_back({1024, 2});

    int inputChannels = outputChannels;
    for (int i = 0; i < convSettings.size(); i++) {
        auto setting   = convSettings[i];
        outputChannels = setting[0];
        int times      = setting[1];
        outputChannels = makeDivisible(outputChannels * widthMult, divisor);

        for (int j = 0; j < times; j++) {
            int stride = 1;
            if (times > 1 && j == 0) {
                stride = 2;
            }

            convBlocks.emplace_back(DepthwiseSeparableConv2D({inputChannels, outputChannels}, stride));
            inputChannels = outputChannels;
        }
    }

    dropout.reset(NN::Dropout(0.1));
    fc.reset(NN::Linear(1024, numClasses, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

    convOption.channel = {1024, 1024};
    convOption.kernelSize = {7, 7};
    convOption.depthwise = true;
    convOption.padMode = SAME;
    convOption.stride = {7, 7};
    conv_ap.reset(NN::Conv(convOption));

    registerModel({conv1, bn1, dropout, fc, conv_ap});
    registerModel(convBlocks);
    setName("MobilenetV1");
}

std::vector<Express::VARP> MobilenetV1::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = _Relu6(x);

    for (int i = 0; i < convBlocks.size(); i++) {
        x = convBlocks[i]->forward(x);
    }

    // global avg pooling
//    x->getInfo()->printShape();
    x = conv_ap->forward(x);

    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});

    x = dropout->forward(x);
    x = fc->forward(x);

    x = _Softmax(x, 1);
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN
#endif