//
//  demoMain.cpp
//  MNN
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include "DemoUnit.hpp"
#include <chrono>
#include <iostream>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <atomic>
#include <thread>

std::atomic<int> overheatCounter(0);
std::atomic<bool> monitor(true);

void readTempFile(int interval_ms){
    while(monitor) {
        std::ifstream ifs("/sys/class/thermal/thermal_zone0/temp", std::ios::in);
        if (!ifs) {
            MNN_PRINT("failed to open temperature file.\n");
            MNN_ASSERT(false);
        }
        float temp;
        ifs >> temp;
        float temp_c = temp / 1000.0f;
        if (temp_c >= 50.0f) {
            overheatCounter++;
        }
        ifs.close();
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}
int main(int argc, const char* argv[]) {
//#ifdef DEBUG_EXECUTION_DETAIL
//    MNN_PRINT("defined DEBUG_EXECUTION_DETAIL\n");
//#endif
//#ifdef PROFILE_EXECUTION_IN_LOG
//    MNN_PRINT("defined PROFILE_EXECUTION_IN_LOG\n");
//#endif
//#ifdef PROFILE_COST_IN_LOG
//    MNN_PRINT("defined PROFILE_COST_IN_LOG\n");
//#endif
//    printf("main\n");
    if (argc < 2) {
        MNN_ERROR("Usage: ./runTrainDemo.out CASENAME [ARGS]\n");
        auto& list = DemoUnitSet::get()->list();
        MNN_PRINT("Valid Case: \n");

        for (auto iter : list) {
            MNN_PRINT("%s\n", iter.first.c_str());
        }
        return 0;
    }
    auto demo = DemoUnitSet::get()->search(argv[1]);
    if (nullptr == demo) {
        MNN_ERROR("Can't find demo %s\n", argv[1]);
        return 0;
    }
    std::thread tempThread(readTempFile, 1000);
    auto start = std::chrono::high_resolution_clock::now();
    demo->run(argc - 1, argv + 1);
    monitor = false;
    tempThread.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Total time: " << duration.count() << " s" << std::endl;
    std::cout << "Overheat count: " << overheatCounter.load() << std::endl;
    return 0;
}
