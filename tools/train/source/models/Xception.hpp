//
//  Xception.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Xception_hpp
#define Xception_hpp

#include <MNN/expr/Module.hpp>
#include "ModelUtils.hpp"

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC Xception : public Express::Module {
public:
    Xception(int numClass=1001);
    std::shared_ptr<Module>entryF, midF, exitF;
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // XceptionModels_hpp
