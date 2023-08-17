//
//  MobilenetV1NoBN.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MobilenetV1NoBN_hpp
#define MobilenetV1NoBN_hpp

#include <vector>
#include "ModelUtilsNoBN.hpp"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NN.hpp>

#include "ModelUtils.hpp"  // for makeDivisible

namespace MNN {
namespace Train {
namespace Model {
using namespace Express;

class MNN_PUBLIC MobilenetV1NoBN : public Express::Module {
public:
    // use tensorflow numClasses = 1001, which label 0 means outlier of the original 1000 classes
    // so you maybe need to add 1 to your true labels, if you are testing with ImageNet dataset
    MobilenetV1NoBN(int numClasses = 1001, float widthMult = 1.0f, int divisor = 8);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Express::Module> conv1;
    std::vector<std::shared_ptr<Express::Module> > convBlocks;
    std::shared_ptr<Express::Module> dropout;
    std::shared_ptr<Express::Module> fc;
    std::shared_ptr<Express::Module> conv_ap;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // MobilenetV1NoBN_hpp
