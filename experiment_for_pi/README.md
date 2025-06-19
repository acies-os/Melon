**现在遇到的问题**

学长，我把build_64，profile_info，和generate_plan都给转换成了非安卓形式。shell看起来没有什么问题，但是在运行generate plan的时候，不知道为何程序陷入了死循环。

**应该怎样运行**
``` bash
    cd <path/to/melon>
    ./schema/generate.sh
    cp -r experiment_for_pi/ build_pi/
    cd build_pi/
    unzip dataset.zip
    ./build_64.sh
    ./profile_info.sh
    ./generate_plan.sh #在这里遇到问题
```

**需要注意的**

您先看一下不改他原来的c++程序能不能编译，要是可以的话看一下能不能运行generate plan。这个程序的入口在 *ExecutionPlanGenerator.hpp* 最下面的 *GeneratePlan* 这个class的run function里