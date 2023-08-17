//
//  Resnet50.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Resnet50_hpp
#define Resnet50_hpp

#include <MNN/expr/Module.hpp>
#include "ModelUtils.hpp"

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC Resnet50 : public Express::Module {
public:
    Resnet50(int numClass=1001);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;

    std::shared_ptr<Module>cbr, conv_mp;
    std::shared_ptr<Module>convBlock1, convBlock2, convBlock3, convBlock4;
    std::shared_ptr<Module>idBlock1_1, idBlock1_2;
    std::shared_ptr<Module>idBlock2_1, idBlock2_2, idBlock2_3;
    std::shared_ptr<Module>idBlock3_1, idBlock3_2, idBlock3_3, idBlock3_4, idBlock3_5;
    std::shared_ptr<Module>idBlock4_1, idBlock4_2;
    std::shared_ptr<Module>conv_ap, fc;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // Resnet50Models_hpp
