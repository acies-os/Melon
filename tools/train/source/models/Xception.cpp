//
//  .cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef Xception_cpp
#define Xception_cpp

#include "Xception.hpp"
#include "Initializer.hpp"
#include <MNN/expr/NN.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {

class Entryflow : public Module {
public:
    Entryflow();
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Express::Module> cbr1, cbr2;
    std::shared_ptr<Express::Module> ds_conv1_1, ds_conv1_2, conv_mp1;
    std::shared_ptr<Express::Module> ds_conv2_1, ds_conv2_2, conv_mp2;
    std::shared_ptr<Express::Module> ds_conv3_1, ds_conv3_2, conv_mp3;
    std::shared_ptr<Express::Module> conv_res1, conv_res2, conv_res3;
    std::shared_ptr<Express::Module> bn_res1, bn_res2, bn_res3;
};

Entryflow::Entryflow() {
    cbr1 = ConvBnRelu({3, 32}, 3, 2);
    cbr2 = ConvBnRelu({32, 64});
    
    NN::ConvOption convOption;
    convOption.padMode    = SAME;
    convOption.depthwise  = false;

    convOption.kernelSize = {1, 1};
    convOption.stride     = {2, 2};
    convOption.channel    = {64, 128};
    conv_res1.reset(NN::Conv(convOption));
    bn_res1.reset(NN::BatchNorm(128));

    convOption.channel    = {128, 256};
    conv_res2.reset(NN::Conv(convOption));
    bn_res2.reset(NN::BatchNorm(256));

    convOption.channel    = {256, 728};
    conv_res3.reset(NN::Conv(convOption));
    bn_res3.reset(NN::BatchNorm(728));

    convOption.depthwise  = true;
    convOption.kernelSize = {3, 3};
    convOption.stride     = {2, 2};

    convOption.channel    = {128, 128};
    conv_mp1.reset(NN::Conv(convOption));

    convOption.channel    = {256, 256};
    conv_mp2.reset(NN::Conv(convOption));

    convOption.channel    = {728, 728};
    conv_mp3.reset(NN::Conv(convOption));

    ds_conv1_1 = DepthwiseSeparableConv2D({64, 128});
    ds_conv1_2 = DepthwiseSeparableConv2D({128, 128});
    ds_conv2_1 = DepthwiseSeparableConv2D({128, 256});
    ds_conv2_2 = DepthwiseSeparableConv2D({256, 256});
    ds_conv3_1 = DepthwiseSeparableConv2D({256, 728});
    ds_conv3_2 = DepthwiseSeparableConv2D({728, 728});

    registerModel({cbr1, cbr2, conv_res1, conv_res2, conv_res3, bn_res1, bn_res2, bn_res3,
                   ds_conv1_1, ds_conv1_2, conv_mp1, ds_conv2_1, ds_conv2_2, conv_mp2, ds_conv3_1, ds_conv3_2, conv_mp3});

}

std::vector<Express::VARP> Entryflow::onForward(const std::vector<Express::VARP> &inputs) {
    VARP x = inputs[0];

    x = cbr1->forward(x);
    x = cbr2->forward(x);

    auto res1 = conv_res1->forward(x);
    res1      = bn_res1->forward(res1);
    auto res2 = conv_res2->forward(res1);
    res2      = bn_res2->forward(res2);
    auto res3 = conv_res3->forward(res2);
    res3      = bn_res3->forward(res3);

    x = ds_conv1_1->forward(x);
    x = _Relu(x);
    x = ds_conv1_2->forward(x);
    x = conv_mp1->forward(x);
    x = x + res1;

    x = _Relu(x);
    x = ds_conv2_1->forward(x);
    x = _Relu(x);
    x = ds_conv2_2->forward(x);
    x = conv_mp2->forward(x);
    x = x + res2;

    x = _Relu(x);
    x = ds_conv3_1->forward(x);
    x = _Relu(x);
    x = ds_conv3_2->forward(x);
    x = conv_mp3->forward(x);
    x = x + res3;
    return {x};
}

class Middleflow : public Module {
public:
    Middleflow();
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Module>ds_conv1, ds_conv2, ds_conv3;
};

Middleflow::Middleflow() {
    ds_conv1 = DepthwiseSeparableConv2D({728, 728});
    ds_conv2 = DepthwiseSeparableConv2D({728, 728});
    ds_conv3 = DepthwiseSeparableConv2D({728, 728});
    registerModel({ds_conv1, ds_conv2, ds_conv3});
}

std::vector<Express::VARP> Middleflow::onForward(const std::vector<Express::VARP> &inputs) {
    VARP x = inputs[0];
    auto res = x;
    x = _Relu(x);
    x = ds_conv1->forward(x);
    x = _Relu(x);
    x = ds_conv2->forward(x);
    x = _Relu(x);
    x = ds_conv3->forward(x);
    x = x + res;
    return {x};
}

class Exitflow : public Module {
public:
    Exitflow(int numClasses=1000);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Module>ds_conv1, ds_conv2, ds_conv3, ds_conv4;
    std::shared_ptr<Module>conv_mp, conv_ap;
    std::shared_ptr<Module>conv_res, bn_res;
    std::shared_ptr<Module>fc;
};

Exitflow::Exitflow(int numClasses) {
    ds_conv1 = DepthwiseSeparableConv2D({728, 728});
    ds_conv2 = DepthwiseSeparableConv2D({728, 1024});
    ds_conv3 = DepthwiseSeparableConv2D({1024, 1536});
    ds_conv4 = DepthwiseSeparableConv2D({1536, 2048});

    NN::ConvOption convOption;
    convOption.padMode    = SAME;
    convOption.depthwise  = false;
    convOption.channel    = {728, 1024};
    convOption.stride     = {2, 2};
    convOption.kernelSize = {1, 1};
    conv_res.reset(NN::Conv(convOption));
    bn_res.reset(NN::BatchNorm(1024));

    convOption.depthwise  = true;
    convOption.kernelSize = {3, 3};
    convOption.stride     = {2, 2};
    convOption.channel    = {1024, 1024};
    conv_mp.reset(NN::Conv(convOption));

    convOption.kernelSize = {10, 10};
    convOption.stride     = {10, 10};
    convOption.channel    = {2048, 2048};
    conv_ap.reset(NN::Conv(convOption));

    fc.reset(NN::Linear(2048, numClasses, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({ds_conv1, ds_conv2, ds_conv3, ds_conv4, conv_mp, conv_ap, conv_res, bn_res, fc});
}

std::vector<Express::VARP> Exitflow::onForward(const std::vector<Express::VARP> &inputs) {
    VARP x = inputs[0];
    auto res = conv_res->forward(x);
    res      = bn_res->forward(res);
    x = _Relu(x);
    x = ds_conv1->forward(x);
    x = _Relu(x);
    x = ds_conv2->forward(x);
    x = conv_mp->forward(x);
    x = x + res;
    x = ds_conv3->forward(x);
    x = _Relu(x);
    x = ds_conv4->forward(x);
    x = _Relu(x);
    x = conv_ap->forward(x);

    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});
    x = fc->forward(x);

    x = _Softmax(x, 1);
    return {x};
}

Xception::Xception(int numClass) {
    entryF = std::shared_ptr<Module>(new Entryflow());
    midF   = std::shared_ptr<Module>(new Middleflow());
    exitF  = std::shared_ptr<Module>(new Exitflow(numClass));
    setName("Xception");
    registerModel({entryF, midF, exitF});
}

std::vector<Express::VARP> Xception::onForward(const std::vector<Express::VARP>& inputs) {
    VARP x = inputs[0];
    x = entryF->forward(x);
    x = midF->forward(x);
    x = exitF->forward(x);
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN
#endif
