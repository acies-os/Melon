//
//  .cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef Resnet50_cpp
#define Resnet50_cpp

#include "Resnet50.hpp"
#include "Initializer.hpp"
#include <MNN/expr/NN.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {

Resnet50::Resnet50(int numClass) {
    cbr = ConvBnRelu({3, 64}, 7, 2);

    NN::ConvOption convOption;
    convOption.padMode    = SAME;
    convOption.depthwise  = true;

    convOption.kernelSize = {3, 3};
    convOption.channel    = {64, 64};
    convOption.stride     = {2, 2};
    conv_mp.reset(NN::Conv(convOption));

    convBlock1 = Resnet50ConvBlock(3, {64, 64, 64}, {64, 64, 256}, 1);
    idBlock1_1 = IdentityBlock(3, {256, 64, 64}, {64, 64, 256});
    idBlock1_2 = IdentityBlock(3, {256, 64, 64}, {64, 64, 256});

    convBlock2 = Resnet50ConvBlock(3, {256, 128, 128}, {128, 128, 512});
    idBlock2_1 = IdentityBlock(3, {512, 128, 128}, {128, 128, 512});
    idBlock2_2 = IdentityBlock(3, {512, 128, 128}, {128, 128, 512});
    idBlock2_3 = IdentityBlock(3, {512, 128, 128}, {128, 128, 512});

    convBlock3 = Resnet50ConvBlock(3, {512, 256, 256}, {256, 256, 1024});
    idBlock3_1 = IdentityBlock(3, {1024, 256, 256}, {256, 256, 1024});
    idBlock3_2 = IdentityBlock(3, {1024, 256, 256}, {256, 256, 1024});
    idBlock3_3 = IdentityBlock(3, {1024, 256, 256}, {256, 256, 1024});
    idBlock3_4 = IdentityBlock(3, {1024, 256, 256}, {256, 256, 1024});
    idBlock3_5 = IdentityBlock(3, {1024, 256, 256}, {256, 256, 1024});

    convBlock4 = Resnet50ConvBlock(3, {1024, 512, 512}, {512, 512, 2048});
    idBlock4_1 = IdentityBlock(3, {2048, 512, 512}, {512, 512, 2048});
    idBlock4_2 = IdentityBlock(3, {2048, 512, 512}, {512, 512, 2048});

    convOption.kernelSize = {7, 7};
    convOption.channel    = {2048, 2048};
    convOption.stride     = {7, 7};
    conv_ap.reset(NN::Conv(convOption));
    fc.reset(NN::Linear(2048, numClass, true, std::shared_ptr<Initializer>(Initializer::MSRA())));
    registerModel({cbr, conv_mp,  convBlock1, convBlock2, convBlock3, convBlock4, idBlock1_1, idBlock1_2, idBlock2_1, idBlock2_2, idBlock2_3,
                   idBlock3_1, idBlock3_2, idBlock3_3, idBlock3_4, idBlock3_5, idBlock4_1, idBlock4_2, conv_ap, fc});
    setName("Resnet50");
}

std::vector<Express::VARP> Resnet50::onForward(const std::vector<Express::VARP>& inputs) {
    VARP x = inputs[0];
    x = cbr->forward(x);
    x = conv_mp->forward(x);

    x = convBlock1->forward(x);
    x = idBlock1_1->forward(x);
    x = idBlock1_2->forward(x);

    x = convBlock2->forward(x);
    x = idBlock2_1->forward(x);
    x = idBlock2_2->forward(x);
    x = idBlock2_3->forward(x);

    x = convBlock3->forward(x);
    x = idBlock3_1->forward(x);
    x = idBlock3_2->forward(x);
    x = idBlock3_3->forward(x);
    x = idBlock3_4->forward(x);
    x = idBlock3_5->forward(x);

    x = convBlock4->forward(x);
    x = idBlock4_1->forward(x);
    x = idBlock4_2->forward(x);

    x = conv_ap->forward(x);
    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});
    x = fc->forward(x);
    x = _Softmax(x, 1);
    return {x};
}


} // namespace Model
} // namespace Train
} // namespace MNN
#endif
