//
//  Resnet18.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Resnet18_hpp
#define Resnet18_hpp

#include <MNN/expr/Module.hpp>
#include "ModelUtils.hpp"

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC Resnet18 : public Express::Module {
public:
    Resnet18(int numClass=1001);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;

};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // Resnet18Models_hpp
