//
// Created by 王启鹏 on 2021/3/23.
//

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "MobilenetV2.hpp"
#include "MobilenetV1.hpp"
#include "GoogLeNet.hpp"
#include "Squeezenet.hpp"
#include "MobilenetV2NoBN.hpp"
#include "MobilenetV1NoBN.hpp"
#include "SqueezenetNoBN.hpp"
#include "Resnet34NoBN.hpp"
#include "Resnet50NoBN.hpp"
#include "Alexnet.hpp"
#include "Lenet.hpp"
#include "Resnet34.hpp"
#include "Xception.hpp"
#include "Resnet50.hpp"
#include "MnistUtils.hpp"
//#include "MobilenetV2Utils.hpp"
#include "MemTimeProfileUtils.cpp"
#include <MNN/expr/NN.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class MemTimeProfile : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        RandomGenerator::generator(17);
        for(int i=0; i<argc; i++) {
            MNN_DEBUG_PRINT("arg[%d] = %s, ", i, argv[i])
        }
        MNN_DEBUG_PRINT("\n");
        if (argc == 5) {
            std::string modelname = argv[1];
            if(modelname == "Lenet") {
                std::string root = argv[2];
                int batchsize = atoi(argv[3]);
                int microBatchsize = atoi(argv[4]);
                std::shared_ptr<Module> model(new Lenet);
                MnistUtils::train(model, root, batchsize, microBatchsize);
                return 0;
            }
        } else if (argc >= 6 && argc <= 10) {
            std::string modelname = argv[1];
            if (modelname == "MobilenetV2" || modelname == "MobilenetV1" || modelname == "Alexnet"
                    || modelname == "Squeezenet" || modelname == "Googlenet" || modelname == "Xception"
                    || modelname == "Resnet50" || modelname == "Resnet34"
                    || modelname == "MobilenetV2NoBN" || modelname == "MobilenetV1NoBN" || modelname == "SqueezenetNoBN"
                    || modelname == "Resnet34NoBN" || modelname == "Resnet50NoBN") {
                std::string trainImagesFolder = argv[2];
                std::string trainImagesTxt = argv[3];
                std::string testImagesFolder = argv[4];
                std::string testImagesTxt = argv[5];
                int batchsize = atoi(argv[6]);
                int microBatchsize = atoi(argv[7]);
                std::string method = "mnn";
                size_t bgt = 5500, bgt_adap = -1;  //bgt == 5500 or 3800
                float prog = 1.0;
                MNN_DEBUG_PRINT("argc = %d\n", argc)
                if (argc >= 9) {
                    method = argv[8];
                }
                if(argc >= 10) {
                    bgt = atoi(argv[9]);
                    if (bgt == 6) {
                        bgt = 3800;
                    } else if (bgt == 8) {
                        bgt = 5500;
                    }
                    if (bgt != 5500 && bgt != 3800) {
                        std::cout << "usage: \n"
                                     "\t./runTrainDemo.out MemTimeProfile Lenet /path/to/unzipped/mnist/data/ BatchSize MicroBatchSize\n"
                                     "\t./runTrainDemo.out MemTimeProfile MODEL path/to/images/ path/to/txt [BatchSize MicroBatchSize method, budget, budget_adaptive, prog]\n"
                                     "\t\tMODEL=[MobilenetV2|Alexnet|Squeezenet|Googlenet]\n"
                                     "\t\toptional params:\n"
                                     "\t\tBatchSize % MicroBatchSize == 0 or default is 8\n"
                                     "\t\tmethod=[resize|profile|cost | read|write | mnn|sublinear|capuchin|swap|ours | adaptive], mnn by default\n"
                                     "\t\tbudget_adaptive budget=[6|8|5500|3800], 8GB device is used by default\n"
                                     "\t\tprog: float value in [0.0, 1.0], 1.0 by default (no adaptiveness)\n";
                        return 0;
                    }
                }
                if(argc >= 11) {
                    bgt_adap = atoi(argv[10]);
                }
                if(argc >= 12) {
                    prog = atof(argv[11]);
                }
                MNN_DEBUG_PRINT("%s\n", method.c_str());
                std::shared_ptr<Module> model;
                int numClass = 10;
                if (modelname == "MobilenetV2") {
                    numClass = 1001;
                    model = std::make_shared<MobilenetV2>(numClass);
                } else if (modelname == "MobilenetV1") {
                    numClass = 1001;
                    model = std::make_shared<MobilenetV1>(numClass);
                } else if (modelname == "Alexnet") {
                    model = std::make_shared<Alexnet>();
                } else if (modelname == "Squeezenet") {
                    model = std::make_shared<Squeezenet>();
                } else if (modelname == "Googlenet") {
                    model = std::make_shared<GoogLenet>();
                } else if (modelname == "Xception") {
                    numClass = 1001;
                    model = std::make_shared<Xception>(numClass);
                } else if (modelname == "Resnet50") {
                    numClass = 1001;
                    model = std::make_shared<Resnet50>(numClass);
                } else if (modelname == "Resnet34") {
                    numClass = 1001;
                    model = std::make_shared<Resnet34>(numClass);
                } else if (modelname == "MobilenetV2NoBN") {
                    numClass = 1001;
                    model = std::make_shared<MobilenetV2NoBN>(numClass);
                } else if (modelname == "MobilenetV1NoBN") {
                    numClass = 1001;
                    model = std::make_shared<MobilenetV1NoBN>(numClass);
                } else if (modelname == "SqueezenetNoBN") {
                    model = std::make_shared<SqueezenetNoBN>();
                } else if (modelname == "Resnet34NoBN") {
                    numClass = 1001;
                    model = std::make_shared<Resnet34NoBN>(numClass);
                } else if (modelname == "Resnet50NoBN") {
                    numClass = 1001;
                    model = std::make_shared<Resnet50NoBN>(numClass);
                }
                MNN_DEBUG_PRINT("method = %s, bgt = %lu, bgt_adap = %lu, prog = %f\n", method.c_str(), bgt, bgt_adap, prog)
                MNN_DEBUG_PRINT("passed last 3 params: %lu, %lu, %.1f\n", bgt, bgt_adap, prog)
                if (method == "adaptive" && bgt_adap == -1) {
                    std::cout << "usage: \n"
                                 "\t./runTrainDemo.out MemTimeProfile Lenet /path/to/unzipped/mnist/data/ BatchSize MicroBatchSize\n"
                                 "\t./runTrainDemo.out MemTimeProfile MODEL path/to/images/ path/to/txt [BatchSize MicroBatchSize method, budget, budget_adaptive, prog]\n"
                                 "\t\tMODEL=[MobilenetV2|Alexnet|Squeezenet|Googlenet]\n"
                                 "\t\tBatchSize % MicroBatchSize == 0 or default is 8\n"
                                 "\t\tmethod=[resize|profile|cost | read|write | mnn|sublinear|capuchin|swap|ours | adaptive]\n"
                                 "\t\tbudget_adaptive budget=[6|8|5500|3800]\n"
                                 "\t\tprog: [0.0, 1.0]\n";
                    return 0;
                }
                MemTimeProfileUtils::train(model, numClass, 1, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt,
                                           batchsize, microBatchsize, method, bgt, bgt_adap, prog);
                return 0;
            }
        }
        std::cout << "usage: \n"
                     "\t./runTrainDemo.out MemTimeProfile Lenet /path/to/unzipped/mnist/data/ BatchSize MicroBatchSize\n"
                     "\t./runTrainDemo.out MemTimeProfile MODEL path/to/images/ path/to/txt [BatchSize MicroBatchSize method, budget, budget_adaptive, prog]\n"
                     "\t\tMODEL=[MobilenetV2|Alexnet|Squeezenet|Googlenet]\n"
                     "\t\tBatchSize % MicroBatchSize == 0 or default is 8\n"
                     "\t\tmethod=[resize|profile|cost | read|write | mnn|sublinear|capuchin|swap|ours | adaptive]\n"
                     "\t\tbudget_adaptive budget=[6|8|5500|3800]\n"
                     "\t\tprog: [0.0, 1.0]\n";
        return 0;

    }
};


DemoUnitSetRegister(MemTimeProfile, "MemTimeProfile");