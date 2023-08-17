//
//  MobilenetV2Utils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MemTimeProfileUtils_CPP
#define MemTimeProfileUtils_CPP

#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <string>
#include <cmath>
#include <iostream>
#include <vector>
#include "DemoUnit.hpp"
#include "MicroSGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "Loss.hpp"
#include "Transformer.hpp"
#include "ImageDataset.hpp"
#include "module/PipelineModule.hpp"
#include "OpGrad.hpp"
#include<fstream>
#include <memory>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
struct block {
    int data[8192];
};
class MemTimeProfileUtils {
public:
    static void train(std::shared_ptr<MNN::Express::Module> model, const int numClasses, const int addToLabel,
                      std::string trainImagesFolder, std::string trainImagesTxt,
                      std::string testImagesFolder, std::string testImagesTxt,
                      const int batchsize = -1, const int microBatchsize = -1,
                      std::string target = "mnn",
                      const size_t memoryBudgetMB = 5500, const size_t adaptiveBudgetMB = 4000, const float progress = 1.0,
                      const int trainQuantDelayEpoch = 10, const int quantBits = 8) {
        MNN_DEBUG_PRINT("last 3 params: %lu, %lu, %.1f\n", memoryBudgetMB, adaptiveBudgetMB, progress)
//        auto modelMap1 = Express::Variable::load("mnist.snapshot1.mnn");
//        auto modelMap2 = Express::Variable::load("mnist.snapshot1.mnn");
//        auto modelMap3 = Express::Variable::load("mnist.snapshot1.mnn");
//        for (int i = 0; i < modelMap1.size(); ++i) {
//            modelMap3[i]->input((modelMap1[i] + modelMap2[i]) / _Scalar<float>(2));
//        }
//        Express::Variable::save(modelMap3, "mnist.snapshot.agg.mnn");
//        return;

        if (target == "read"){
            AUTOTIME;
            size_t cnt = 0;
            for (int i = 0; i < 100; ++i) {
                std::ifstream ifs;
                ifs.open("test-write.bat", std::ios::in | std::ios::binary);
                void *ptr = (void *) malloc(1024);
                size_t read_count = 0;
                while (ifs.read((char *) ptr, 1024)) {
                    size_t readedBytes = ifs.gcount();
                    read_count += readedBytes;
                }
                MNN_PRINT("%d: read %lu Bytes\n", i, read_count);
                cnt += read_count;
                ifs.close();
            }
            MNN_PRINT("total %lu Bytes\n", cnt);
            return;
        } else if (target == "write") {
            AUTOTIME;
            size_t cnt = 0;
            size_t fsize = 1024 * 1024 * 1024;
            void *ptr = (void *) malloc(fsize);
            for (int i = 0; i < 10; ++i) {
                FILE *f = fopen("test-write.bat", "wb");
                size_t numwrite = -1;
                if (f != nullptr) {
                    numwrite = fwrite(ptr, sizeof(char), fsize, f);
                    cnt += numwrite;
                }
            }
            MNN_PRINT("total %lu Bytes\n", cnt);
            return;
        } else if (target == "memcpy") {
            size_t fsize = 128 * 1024 * 1024;
            int *ptr = (int *) malloc(fsize * sizeof(int));
            for (size_t i = 0; i < fsize; i++) {
                ptr[i] = i;
            }
            int *new_ptr = (int *) malloc(fsize * sizeof(int));
            AUTOTIME;
            for(int i=0; i<100; i++) {
                if (i % 2 == 0) {
                    memcpy(new_ptr, ptr, fsize * sizeof(int));
                } else {
                    memcpy(ptr, new_ptr, fsize * sizeof(int));
                }

            }
            return ;
        }


        auto exe = Executor::getGlobalExecutor();
        BackendConfig config;
//    std::shared_ptr<SGD> solver(new SGD(model));

        int trainBatchSize, trainMicroBatchsize;
        int trainNumWorkers = 1;
        int testBatchSize = 1;
        int testNumWorkers = 0;
        if (batchsize != -1 && microBatchsize != -1 && batchsize % microBatchsize == 0) {
            trainBatchSize = batchsize;
            trainMicroBatchsize = microBatchsize;
        } else {
            trainBatchSize = trainMicroBatchsize = 8;
        }
#ifdef MNN_OPENCL
        MNN_DEBUG_PRINT("forward type: MNN_FORWARD_OPENCL\n")
        exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 4);
        exe->configExecution(model->name() + "_CL", trainMicroBatchsize, target, memoryBudgetMB, adaptiveBudgetMB, progress);
        MNN_PRINT("model = %s_CL, batch = %d, budget = %d, adapBUdget = %d, progress = %.1f, target = %s\n",
                  model->name().c_str(), trainMicroBatchsize, memoryBudgetMB, adaptiveBudgetMB, progress, target.c_str())

#else
        exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 4);
        exe->configExecution(model->name(), trainMicroBatchsize, target, memoryBudgetMB, adaptiveBudgetMB, progress);
        MNN_DEBUG_PRINT("model = %s, batch = %d, budget = %lu, adapBudget = %lu, progress = %.1f, target = %s\n",
                  model->name().c_str(), trainMicroBatchsize, memoryBudgetMB, adaptiveBudgetMB, progress, target.c_str())
#endif

        std::shared_ptr<MicroSGD> solver(new MicroSGD(model, trainBatchSize, trainMicroBatchsize));
        solver->setMomentum(0.9f);
        // solver->setMomentum2(0.99f);
        solver->setHeuristicFlag(target == "ours");
        solver->setWeightDecay(0.00004f);

        auto converImagesToFormat  = CV::RGB;
        int resizeHeight           = 224;
        int resizeWidth            = 224;
        if (model->name() == "Xception") {
            resizeHeight = resizeWidth = 299;
        }
        std::vector<float> means = {127.5, 127.5, 127.5};
        std::vector<float> scales = {1/127.5, 1/127.5, 1/127.5};
        std::vector<float> cropFraction = {0.875, 0.875}; // center crop fraction for height and width
        bool centerOrRandomCrop = false; // true for random crop
        std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(
                ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means,cropFraction, centerOrRandomCrop));
        bool readAllImagesToMemory = false;
        auto trainDataset = ImageDataset::create(trainImagesFolder, trainImagesTxt, datasetConfig.get(), readAllImagesToMemory);
        auto testDataset = ImageDataset::create(testImagesFolder, testImagesTxt, datasetConfig.get(), readAllImagesToMemory);

        auto trainDataLoader = trainDataset.createLoader(trainMicroBatchsize, true, true, trainNumWorkers);
        auto testDataLoader = testDataset.createLoader(testBatchSize, true, true, testNumWorkers);

        const int trainIterations = trainDataLoader->iterNumber();
        const int testIterations = testDataLoader->iterNumber();

        for (int epoch = 0; epoch < 50; ++epoch) {
            model->clearCache();
            exe->gc(Executor::FULL);
            exe->resetProfile();
            {
                AUTOTIME;
                trainDataLoader->reset();
                model->setIsTraining(true);
                // turn float model to quantize-aware-training model after a delay
                if (epoch == trainQuantDelayEpoch) {
                    // turn model to train quant model
                    std::static_pointer_cast<PipelineModule>(model)->toTrainQuant(quantBits);
                }
                int numIteration = 2;
                if (target == "adaptive") numIteration = 1;
                if (target == "profile" || target == "resize" || target == "cost") {
                    numIteration = 1;
                }
                for (int i = 0; i < numIteration * (trainBatchSize / trainMicroBatchsize); i++) {
                    MNN_DEBUG_PRINT("exe->gc()\n")
                    exe->gc();
                    trainDataLoader->reset();
                    AUTOTIME;
                    MNN_MEMORY_PROFILE("begin an iteration")
                    MNN_PRINT("begin an iteration %d\n", i);
                    auto trainData  = trainDataLoader->next();
                    auto example    = trainData[0];

                    // Compute One-Hot
                    auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {1})),
                                             _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                             _Scalar<float>(0.0f));

                    auto predict = model->forward(_Convert(example.first[0], NC4HW4));
                    auto loss    = _CrossEntropy(predict, newTarget);
                    // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                    float rate = 1e-5;
//                    loss->readMap<float>();
//                    return;
                    solver->setLearningRate(rate);
//                if (solver->currentStep() % 10 == 0) {
//                    std::cout << "train iteration: " << solver->currentStep();
//                    std::cout << " loss: " << loss->readMap<float>()[0];
//                    std::cout << " lr: " << rate << std::endl;
//                }
                    solver->step(loss);
//                    loss->readMap<float>();
//                    return;
                }
                return;
            }

            int correct = 0;
            int sampleCount = 0;
            testDataLoader->reset();
            model->setIsTraining(false);
            exe->gc(Executor::PART);

            AUTOTIME;
            for (int i = 0; i < testIterations; i++) {
                auto data       = testDataLoader->next();
                auto example    = data[0];
                auto predict    = model->forward(_Convert(example.first[0], NC4HW4));
                predict         = _ArgMax(predict, 1); // (N, numClasses) --> (N)
                auto label = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
                sampleCount += label->getInfo()->size;
                auto accu       = _Cast<int32_t>(_Equal(predict, label).sum({}));
                correct += accu->readMap<int32_t>()[0];

                if ((i + 1) % 10 == 0) {
                    std::cout << "test iteration: " << (i + 1) << " ";
                    std::cout << "acc: " << correct << "/" << sampleCount << " = " << float(correct) / sampleCount * 100 << "%";
                    std::cout << std::endl;
                }
            }
            auto accu = (float)correct / testDataLoader->size();
            // auto accu = (float)correct / usedSize;
            std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;

            {
                auto forwardInput = _Input({1, 3, resizeHeight, resizeWidth}, NC4HW4);
                forwardInput->setName("data");
                auto predict = model->forward(forwardInput);
                Transformer::turnModelToInfer()->onExecute({predict});
                predict->setName("prob");
                Variable::save({predict}, "temp.mobilenetv2.mnn");
            }

            exe->dumpProfile();
        }
    }
};

#endif
