//
//  Executor.cpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <set>
#include <MNN/expr/Executor.hpp>
#include <unistd.h>
#include "MNN/MNNDefine.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "Utils.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "core/WrapExecution.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include <core/BufferAllocator.hpp>
#include <iostream>
#ifdef MNN_EXPR_ENABLE_PROFILER
#define MNN_EXPRESS_ERROR_REPORT
#endif
#define MNN_EXPRESS_OPEN_MEMORY_REUSE
//#define ALLOCATE_CACHE_ID_RUNTIME
//#define PROFILE_EXECUTION_IN_LOG
//#define DEBUG_EXECUTION_DETAIL
//#define PROFILE_COST_IN_LOG
//0: memory_pool    1: OS   2: hybrid
int dynamic_type = 0;

//#ifdef DEBUG_EXECUTION_DETAIL
//#define MNN_DEBUG_PRINT(format, ...) MNN_PRINT(format, ##__VA_ARGS__)
//#else
//#define MNN_DEBUG_PRINT(format, ...)
//#endif


namespace MNN {
namespace Express {
#ifdef MNN_EXPR_ENABLE_PROFILER
class Executor::Profiler {
public:
    void reset();
    void dump() const;
    void add(const std::string& opType, float timeInMs);
    void addFlops(const std::string& opType, float flops);
private:
    std::map<std::string, float> mTimes;
    std::map<std::string, float> mFlops;
};
void Executor::Profiler::reset() {
    mTimes.clear();
    mFlops.clear();
}
void Executor::Profiler::dump() const {
    float sumValue = 0.0f;
    for (auto iter : mTimes) {
        MNN_PRINT("%s: %f ms\n", iter.first.c_str(), iter.second);
        sumValue += iter.second;
    }
    MNN_PRINT("Total: %f ms\n", sumValue);
    sumValue = 0.0f;
    for (auto iter : mFlops) {
        MNN_PRINT("%s: %f \n", iter.first.c_str(), iter.second);
        sumValue += iter.second;
    }
    MNN_PRINT("Total flops: %f M\n", sumValue);
}
void Executor::Profiler::add(const std::string& opType, float timeInMs) {
    auto iter = mTimes.find(opType);
    if (iter == mTimes.end()) {
        mTimes[opType] = timeInMs;
        return;
    }
    iter->second += timeInMs;
}
void Executor::Profiler::addFlops(const std::string& opType, float flops) {
    auto iter = mFlops.find(opType);
    if (iter == mFlops.end()) {
        mFlops[opType] = flops;
        return;
    }
    iter->second += flops;
}
#endif
void Executor::setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread) {
    std::lock_guard<std::mutex> _l(mMutex);
    auto creator = MNNGetExtraRuntimeCreator(type);
    if (nullptr == creator) {
        MNN_ERROR("Error to find creator of %d, set CPU default\n", type);
        type = MNN_FORWARD_CPU;
        creator = MNNGetExtraRuntimeCreator(type);
    }
    MNN_ASSERT(nullptr != creator);
    Backend::Info info;
    info.type = type;
    info.mode = Backend::Info::DIRECT;
    info.numThread = numberThread;
    info.user = (BackendConfig*)&config;
    std::shared_ptr<Runtime> bn(creator->onCreate(info));
    mRuntime.first = bn;
    mRuntime.second = type;
}

void Executor::gc(GCFlag flag) {
    if (FULL == flag) {
        mBackupRuntime.first->onGabageCollect(100);
        mRuntime.first->onGabageCollect(100);
    } else {
        mBackupRuntime.first->onGabageCollect(0);
        mRuntime.first->onGabageCollect(0);
    }
}
Executor::Executor(std::shared_ptr<Runtime> backend, MNNForwardType type) {
    mRuntime.first = backend;
    mRuntime.second = type;
    Backend::Info info;
    info.type = MNN_FORWARD_CPU;
    auto cre = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
    info.mode = Backend::Info::DIRECT;
    info.numThread = 1;
    mBackupRuntime.first.reset(cre->onCreate(info));
    mBackupRuntime.second = MNN_FORWARD_CPU;

#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler.reset(new Profiler);
#endif
}
Executor::~Executor(){
    mRuntime.first = nullptr;
    mBackupRuntime.first = nullptr;
}
Executor::Requirement Executor::getRequirement(Expr* expr) const {
    Executor::Requirement req;
    auto op = expr->get();
    auto inputSize = expr->inputs().size();
    req.contentNeedContent.resize(inputSize);
    req.shapeNeedContent.resize(inputSize);
    if (op->type() == OpType_Extra) {
        for (int i = 0; i < inputSize; ++i) {
            req.contentNeedContent[i] = true;
            req.shapeNeedContent[i]   = false;
        }
        return req;
    }
    for (int i = 0; i < inputSize; ++i) {
        req.contentNeedContent[i] = SizeComputer::opNeedContent(op->type(), i);
        req.shapeNeedContent[i]   = false;
    }
    auto needIndexId = SizeComputer::needInputContent(op);
    for (auto index : needIndexId) {
        if (index < req.shapeNeedContent.size()) {
            req.shapeNeedContent[index] = true;
        }
    }
    return req;
}

static std::once_flag gInitFlag;
std::shared_ptr<Executor> Executor::getGlobalExecutor() {
    static std::shared_ptr<Executor> gExecutor;
    std::call_once(gInitFlag, [&]() {
        auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
        Backend::Info info;
        info.type = MNN_FORWARD_CPU;
        info.numThread = 1;
        std::shared_ptr<Runtime> bn(creator->onCreate(info));
        gExecutor.reset(new Executor(bn, MNN_FORWARD_CPU));
    });
    return gExecutor;
}

std::shared_ptr<Executor> Executor::newExecutor(MNNForwardType type,
                                                const BackendConfig& config,
                                                int numberThread) {
    auto creator = MNNGetExtraRuntimeCreator(type);
    Backend::Info info;
    info.type = type;
    info.numThread = numberThread;
    std::shared_ptr<Runtime> bn(creator->onCreate(info));
    return std::shared_ptr<Executor>(new Executor(bn, type));
}

RuntimeInfo Executor::getRuntime() {
    RuntimeInfo info;
    auto glo = ExecutorScope::Current();
    info.second = glo->mBackupRuntime.first;
    info.first.insert(std::make_pair(glo->mRuntime.second, glo->mRuntime.first));
    return info;
}

ErrorCode Executor::computeInfo(Expr* expr) {
    MNN_ASSERT(nullptr != expr);
    MNN_ASSERT(nullptr != expr->get());
    if (expr->get()->type() == OpType_Extra) {
        return NOT_SUPPORT;
    }
    auto op = expr->get();
    std::vector<Tensor*> inputTensors(expr->inputs().size());
    for (int i=0; i<inputTensors.size(); ++i) {
        auto inputExpr = expr->inputs()[i]->expr();
        inputTensors[i] = inputExpr.first->inside()->mOutputTensors[inputExpr.second];
    }
    bool res = SizeComputer::computeOutputSize(op, inputTensors, expr->inside()->mOutputTensors);
    if (!res) {
        // Compute Error
#ifdef MNN_EXPRESS_ERROR_REPORT
        if (expr->name().empty()) {
            MNN_ERROR("Error to compute shape for %s\n", EnumNameOpType(op->type()));
        } else {
            MNN_ERROR("Error to compute shape for %s, %s\n", EnumNameOpType(op->type()), expr->name().c_str());
        }
#endif
        return COMPUTE_SIZE_ERROR;
    }
    for (int i = 0; i < expr->outputSize(); ++i) {
        auto tensor = expr->inside()->mOutputTensors[i];
        TensorUtils::setLinearLayout(tensor);
        auto shape  = expr->outputInfo(i);
        Utils::copyTensorToInfo(shape, tensor);
    }
    return NO_ERROR;
}
class Executor::ComputeCache {
public:
    void setShapeDirty();
    void setContentDirty();
    void* mapOutput(int offset, Tensor* dest);

    ~ ComputeCache();
    ComputeCache(std::shared_ptr<Backend> backend, std::shared_ptr<Backend> backupBackend);

    ErrorCode compute();
    ErrorCode resize();
    ErrorCode computeDirectly();
    ErrorCode computeViaCheckpoint();
    ErrorCode computeViaStrategy();
    ErrorCode computeAdaptively();
    ErrorCode computeViaSwapping();
    ErrorCode computeIthOp(int i, bool profile=false, bool recompute=false, std::vector<int> skipReleaseOpID={}, bool viaStrategy=false, bool enableSwap=false);
    ErrorCode setExecutionStrategy(std::string model, int batch, int bgt);
    void setMethodAndTarget(std::string method, std::string target);
    void config(std::string model, int batch);
    void setBudgetAndProgress(size_t bgt, size_t bgt_adap, float adaptiveProgress);

private:
    std::set<std::shared_ptr<ComputeCache>> mInputs;
    std::vector<Tensor*> mOutputs;
    std::vector<std::shared_ptr<Unit>> mUnits;
    std::shared_ptr<Backend> mBackend;
    std::shared_ptr<Backend> mBackupBackend;
    std::set<std::shared_ptr<Expr::Inside>> mInputInside;
    friend class Executor;
    bool mContentDirty = true;
    bool mShapeDirty = true;
    GeometryComputer::Context mContext;
    CommandBuffer mCmdBuffer;
    std::vector<std::shared_ptr<Execution>> mExecutions;
    std::map<const Op*, std::shared_ptr<Execution>> mCacheExes;
    std::vector<std::pair<std::string, int>> mExecuteStrategy;
    bool mComputeHeuristically=false;

    bool zeroInputs() {
//        return mInputs.empty();
        return true;
    }
    static ErrorCode swapout(const Tensor* tensor);
    static ErrorCode swapin(const Tensor* tensor);
    ErrorCode profileExecution();
    int mUniqueCacheID = 0;
    std::vector<bool> opNeedRecompute;
    static std::vector<std::set<int>> opInputs;  // i-th op's inputs and outputs tensors' ids
//    static std::vector<std::set<int>> opGraph, reversedOpGraph;  // simulate input dependency between op-op (adjacency list)
    static std::vector<std::vector<int>> fpLevelList;
    static std::vector<int> tensorFromOp;  // tensorID -> opID
    static std::vector<int> opLevel;
    static std::vector<int> validCkeckpoints;
    static int profiledExecutionSizeWithValidCheckpoint;
    static void getValidCheckpointLevel();
    static std::vector<int> getComputeSequence(const char* filename = nullptr);
    static std::vector<int> selectCheckpoint(int cnt=1);
    std::vector<int> getRecomputeOpList(int curOpID);
    std::string mModelname = "";
    int mBatchsize = 0;
    std::string mComputeMethod = "direct";  // == "direct" || "sublinear" || "strategy"
    std::string mComputeTarget = "mnn";  // == "mnn" || "sublinear" || "ours" || "capuchin" || "profile" || "resize" || "cost"
    std::set<Tensor*> allocatedTensor;
    std::vector<int> featureMap;
    std::map<int, bool>featureSwapoutFlag;
    size_t budget, adaptiveBudget;
    float adaptiveProgress;
};
std::vector<std::set<int>> Executor::ComputeCache::opInputs;  // i-th op's inputs and outputs tensors' ids
//std::vector<std::set<int>> Executor::ComputeCache::opGraph, Executor::ComputeCache::reversedOpGraph;  // simulate input dependency between op-op (adjacency list)
std::vector<int> Executor::ComputeCache::tensorFromOp;  // tensorID -> opID
std::vector<int> Executor::ComputeCache::opLevel;
std::vector<int> Executor::ComputeCache::validCkeckpoints;
std::vector<std::vector<int>> Executor::ComputeCache::fpLevelList;
int Executor::ComputeCache::profiledExecutionSizeWithValidCheckpoint = 0;
void Executor::setShapeDirty(ComputeCache* cache) {
    cache->setShapeDirty();
}
void Executor::setContentDirty(ComputeCache* cache) {
    cache->setContentDirty();
}
void* Executor::mapOutput(ComputeCache* cache, int offset, Tensor* dest) {
    return cache->mapOutput(offset, dest);
}

struct Executor::Unit {
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    const Op* op;
    std::weak_ptr<Expr::Inside> inside;
    std::vector<std::shared_ptr<Tensor>> outputContents;
};
Tensor* Executor::getOutput(ComputeCache* cache, int offset) {
    return cache->mOutputs[offset];
}

void* Executor::ComputeCache::mapOutput(int offset, Tensor* dest) {
    auto tensor = mOutputs[offset];
    if (0 == tensor->deviceId()) {
        auto ptr =  tensor->host<void>();
        Utils::releaseMemoryForHostTensor(dest);
        TensorUtils::getDescribe(dest)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        dest->buffer().host = (uint8_t*)ptr;
        //MNN_ASSERT(nullptr != ptr);
        return ptr;
    }
    Utils::allocMemoryForHostTensor(dest);
    tensor->copyToHostTensor(dest);
    MNN_ASSERT(nullptr != dest->host<void>());
    return dest->host<void>();
}

void Executor::ComputeCache::setShapeDirty() {
    mShapeDirty = true;
}

void Executor::ComputeCache::setContentDirty() {
    mContentDirty = true;
}

Executor::ComputeCache::ComputeCache(std::shared_ptr<Backend> backend, std::shared_ptr<Backend> backupBackend) : mContext(backupBackend) {
    mBackend = backend;
    mBackupBackend = backupBackend;
}
Executor::ComputeCache::~ComputeCache() {
    MNN_DEBUG_PRINT("call %s\n", __FUNCTION__ )
    mUnits.clear();
    mCacheExes.clear();
}
ErrorCode Executor::ComputeCache::compute() {
    MNN_DEBUG_PRINT("call ComputeCache::compute\n")
    allocatedTensor.clear();
//    MNN_ASSERT(validCkptLevel.size()==0)
    if (mShapeDirty) { // default true
        auto code = resize();
        if (zeroInputs()) {
            mExecutions.resize(mCmdBuffer.command.size());
        }
        if (NO_ERROR != code) {
            return code;
        }
    }
    if (!mContentDirty) {
        return NO_ERROR;
    }
    /*在MNN训练的设定里面这两个for都没有什么实际的意义……T^T*/
    for (auto& c : mInputInside) {
        if (c->mContentDirty) {
            // InputType = VARP::INPUT
            return CALL_BACK_STOP;
        }
    }
    for (auto c : mInputs) {
        auto code = c->compute();
        if (NO_ERROR != code) {
            return code;
        }
    }
    //只有在需要计算的时候才能clear buffer，不然会报错
//    mBackend->onClearBuffer();
//    MNN_DEBUG_PRINT("%s: finish onClearBuffer\n", __FUNCTION__ )
//    ExecutorScope::Current()->gc();
    MNN_DEBUG_PRINT("before execute %lu cmds\n", mExecutions.size())
    MNN_MEMORY_PROFILE("before execute %lu cmds", mExecutions.size())
    tensorFromOp.resize(mExecutions.size());
    opNeedRecompute.resize(mExecutions.size());
    ErrorCode code;
    if (mComputeMethod == "direct") {
        MNN_DEBUG_PRINT("call computeDirectly due to mComputeMethod==direct\n")
        code = computeDirectly();
    } else if (mComputeMethod == "sublinear") {
        MNN_DEBUG_PRINT("call computeViaCheckpoint due to mComputeMethod==sublinear\n")
        code = computeViaCheckpoint();
    } else if (mComputeMethod == "strategy") {
        if(mExecuteStrategy.empty()) {
            MNN_PRINT("call computeDirectly due to mComputeMethod==strategy but mExecuteStrategy is empty\n")
            code = computeDirectly();
        } else if (adaptiveBudget != -1 && adaptiveProgress < 1.0) {
            MNN_DEBUG_PRINT("call computeAdaptively\n")
            code = computeAdaptively();
        } else {
            MNN_DEBUG_PRINT("call computeViaStrategy\n")
            code = computeViaStrategy();
        }
    } else if (mComputeMethod == "swap") {
        MNN_DEBUG_PRINT("call computeViaSwapping due to mComputeMethod==swap\n")
        code = computeViaSwapping();
    } else if (mComputeMethod == "adaptive") {
        MNN_DEBUG_PRINT("call computeAdaptively due to mComputeMethod==adaptive\n")
        code = computeAdaptively();
    } else {
        MNN_ASSERT(0)
        return COMPUTE_METHOD_ERROR;
    }

    if (code != NO_ERROR) {
        return code;
    }

    mContentDirty = false;
    MNN_DEBUG_PRINT("finish %s return no_error\n", __FUNCTION__ )
    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::computeDirectly() {
    MNN_DEBUG_PRINT("call %s\n", __FUNCTION__ );
    mBackend->onExecuteBegin();
    mBackupBackend->onExecuteBegin();
    MNN_DEBUG_PRINT("mExecutions.size() = %lu\n", mCmdBuffer.command.size());
#ifndef ALLOCATE_CACHE_ID_RUNTIME
    for(int i=0; i<mCmdBuffer.command.size(); i++) {
        MNN_ASSERT(mCmdBuffer.command[i].outputs.size() == 1)
        for(auto output: mCmdBuffer.command[i].outputs) {
            output->setCacheID(mUniqueCacheID++);
            tensorFromOp[output->cacheID()] = i;
        }
    }
#endif
    MNN_ASSERT(mExecutions.size() == mCmdBuffer.command.size());
    MNN_DEBUG_PRINT("%s: start compute %lu cmds\n", __FUNCTION__, mCmdBuffer.command.size());
    for (int i=0; i<mCmdBuffer.command.size(); ++i) {
//        MNN_DEBUG_PRINT("start compute cmd[%d]:\n", i)
//        AUTOTIME;
        ErrorCode code = computeIthOp(i);
        if (code != NO_ERROR) {
            return code;
        }
    }
    mBackend->onExecuteEnd();
    mBackupBackend->onExecuteEnd();
//    mBackend->onClearBuffer();
//    mBackupBackend->onClearBuffer();

    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::computeViaSwapping() {
    MNN_DEBUG_PRINT("call %s\n", __FUNCTION__ );
    char filename[100];
    sprintf(filename, "vdnn/%s.featuremap.out", mModelname.c_str());
    std::ifstream ifs(filename);
    int a;
    featureMap.clear();
    std::map<int, size_t> tensorSize;
    while(ifs >> a) {
        featureMap.push_back(a);
    }
    ifs.close();
    mBackend->onExecuteBegin();
    mBackupBackend->onExecuteBegin();
    MNN_DEBUG_PRINT("mExecutions.size() = %lu\n", mCmdBuffer.command.size());
#ifndef ALLOCATE_CACHE_ID_RUNTIME
    for(int i=0; i<mCmdBuffer.command.size(); i++) {
        MNN_ASSERT(mCmdBuffer.command[i].outputs.size() == 1)
        for(auto output: mCmdBuffer.command[i].outputs) {
            output->setCacheID(mUniqueCacheID++);
            tensorFromOp[output->cacheID()] = i;
            tensorSize[output->cacheID()] = output->size();
        }
    }
#endif
    std::sort(featureMap.begin(), featureMap.end(), [&tensorSize](int p, int q) {return tensorSize[q] > tensorSize[q];});
    MNN_ASSERT(mExecutions.size() == mCmdBuffer.command.size());
    MNN_DEBUG_PRINT("%s: start compute %lu cmds\n", __FUNCTION__, mCmdBuffer.command.size());
    for (int i=0; i<mCmdBuffer.command.size(); ++i) {
//        MNN_DEBUG_PRINT("start compute cmd[%d]:\n", i)
//        AUTOTIME;
        ErrorCode code = computeIthOp(i, false, false, {}, false, true);
        if (code != NO_ERROR) {
            return code;
        }
    }
    mBackend->onExecuteEnd();
    mBackupBackend->onExecuteEnd();
//    mBackend->onClearBuffer();
//    mBackupBackend->onClearBuffer();

    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::computeAdaptively() {
    if (adaptiveProgress < 0) adaptiveProgress = 0;
    if (adaptiveProgress >= 1 || budget <= adaptiveBudget) {
        return computeDirectly();
    }
    MNN_PRINT("call %s: %lu - %lu\n", __FUNCTION__, mExecutions.size(), mExecuteStrategy.size())
    mBackend->onExecuteBegin();
    mBackupBackend->onExecuteBegin();
#ifndef ALLOCATE_CACHE_ID_RUNTIME
    for(int i=0; i<mCmdBuffer.command.size(); i++) {
        for(auto output: mCmdBuffer.command[i].outputs) {
            output->setCacheID(mUniqueCacheID++);
            tensorFromOp[output->cacheID()] = i;
        }
    }
#endif
    int max_computed = -1;
    int progress_len = int(mExecuteStrategy.size() * adaptiveProgress);
    //受限按照 strategy 计算一部分，这部分的computeIthOp不会release所以要手动更新allocatedTensor.erase(t);
    for (int i=0; i<progress_len; i++) {
        auto iter = mExecuteStrategy[i];
        MNN_DEBUG_PRINT("Strategy: %s\t%d\n", iter.first.c_str(), iter.second)
        if (iter.first == "compute" || iter.first == "recompute") {
            if (iter.second >= mExecutions.size()) {
                continue;
            }
            max_computed = std::max(iter.second, max_computed);
            computeIthOp(iter.second, false, false, {}, true);
        } else {
            auto& cmd = mCmdBuffer.command[iter.second];
            for(auto t: cmd.outputs) {
                if (dynamic_type == 0) {
                    mBackend->onReleaseBuffer(t, Backend::DYNAMIC);
                } else if (dynamic_type == 1) {
                    mBackend->onFreeBufferToOS(t);
                } else if (dynamic_type == 2) {
                    mBackend->onFreeBufferHybrid(t);
                } else {
                    MNN_ASSERT(false)
                }
                allocatedTensor.erase(t);
            }
        }
    }
    MNN_DEBUG_PRINT("finish prev-progress in %s\n", __FUNCTION__)
    char filename[100];
    sprintf(filename, "heuristic/execution/%s/%s.%d.%lu.execution.txt", mModelname.c_str(), mModelname.c_str(), mBatchsize, adaptiveBudget);
    std::ifstream ifs(filename, std::ios::in);
    std::string str;
    int a;
    mExecuteStrategy.clear();
    std::set<int> tensorIDShouldAppear;
    //加载新的ExecutionStrategy
    while (ifs >> str >> a) {
        if (a <= max_computed) {
            if (str.find("compute") != std::string::npos) {
                tensorIDShouldAppear.insert(a);
            } else {
                tensorIDShouldAppear.erase(a);
            }
        } else {
            mExecuteStrategy.push_back(std::make_pair(str, a));
        }
    }
    ifs.close();
    //保留Tensor的交集，确定重新计算的集合
    std::set<int> allocatedTensorID;
    for(auto t: allocatedTensor) {
        allocatedTensorID.insert(t->cacheID());
    }
    std::set<int> tensorIDReversedBeforeShrink;
    std::set_intersection(tensorIDShouldAppear.begin(), tensorIDShouldAppear.end(),
                          allocatedTensorID.begin(), allocatedTensorID.end(), inserter(tensorIDReversedBeforeShrink, tensorIDReversedBeforeShrink.begin())); // tensorIDReversedBeforeshrink
    std::vector<Tensor*> tensorReversedBeforeShrink;
    for(auto t: allocatedTensor) {
        if (tensorIDReversedBeforeShrink.find(t->cacheID()) != tensorIDReversedBeforeShrink.end()) {
            tensorReversedBeforeShrink.push_back(t);
        }
    }
    auto tensorReversedAfterShrink = mBackend->moveTensor2bottom(tensorReversedBeforeShrink, adaptiveBudget);
    if (tensorReversedAfterShrink.empty()) {
        tensorReversedAfterShrink = mBackupBackend->moveTensor2bottom(tensorReversedBeforeShrink, adaptiveBudget);
    }
    tensorIDReversedBeforeShrink.clear();

    std::vector<int> tensorIDNeedRecompute;
    std::vector<int> tensorIDReversedAfterShrink;
    for (auto t: tensorReversedAfterShrink) {
        tensorIDReversedAfterShrink.push_back(t->cacheID());
    }
    std::set_difference(tensorIDShouldAppear.begin(), tensorIDShouldAppear.end(),
                        tensorIDReversedAfterShrink.begin(), tensorIDReversedAfterShrink.end(), inserter(tensorIDNeedRecompute, tensorIDNeedRecompute.begin()));
    std::sort(tensorIDNeedRecompute.begin(), tensorIDNeedRecompute.end());
    //按照MNN原始的计算方式重新计算: 更新引用计数， recompute的方recompute的方式计算
    for (auto k: tensorIDNeedRecompute) {
        auto& cmd_k = mCmdBuffer.command[k];
        auto op_k = cmd_k.op;
        if (!cmd_k.buffer.empty()) {
            op_k = flatbuffers::GetMutableRoot<Op>(cmd_k.buffer.data());
        }
        for (auto v = 0; v<cmd_k.inputs.size(); ++v) {
            if (!SizeComputer::opNeedContent(op_k->type(), v)) {
                continue;
            }
            auto des = TensorUtils::getDescribe(cmd_k.inputs[v]);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
                des->useCount+=1;
                continue;
            }
            for (auto& s : des->regions) {
                auto subDes = TensorUtils::getDescribe(s.origin);
                if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                    subDes->useCount+=1;
                }
            }
        }
    }
    MNN_DEBUG_PRINT("op need recompute in adaptive: ")
    for (auto k: tensorIDNeedRecompute) {
        MNN_DEBUG_PRINT("%d, ", k)
    }
    MNN_DEBUG_PRINT("\n")
    for (auto k: tensorIDNeedRecompute) {
        MNN_DEBUG_PRINT("start recompute execution[%d]\n", k)
        computeIthOp(k, false, true);
        MNN_DEBUG_PRINT("\tfinish recompute execution[%d]\n", k);
    }
    //加载新的allocation Plan
    MNN_DEBUG_PRINT("%s: loading new plan\n", __FUNCTION__)
    mBackend->setHeuristicStrategy(true, mModelname, mBatchsize, adaptiveBudget, false, false);
    mBackupBackend->setHeuristicStrategy(true, mModelname, mBatchsize, adaptiveBudget, false, false);
    MNN_DEBUG_PRINT("%s: moving tensors to new location\n", __FUNCTION__)
    //将tensor按照新的地址进行排布，因为重新计算的过程和新的strategy不同，所以allocatedTensor包含了shouldAppear以及一些额外的tensor需要剔除
    std::vector<Tensor*> tensorNeedMove;
    for(auto t: allocatedTensor) {
        if(tensorIDShouldAppear.find(t->cacheID()) != tensorIDShouldAppear.end()) {
            tensorNeedMove.push_back(t);
        } else {
            TensorUtils::getDescribe(t)->backend->onReleaseBuffer(t, Backend::DYNAMIC);
        }
    }
    if(!mBackend->adaptTensorToNewAddress(tensorNeedMove)) {
        mBackupBackend->adaptTensorToNewAddress(tensorNeedMove);
    }
    MNN_DEBUG_PRINT("%s: start training via new plan\n", __FUNCTION__)

    for (auto iter: mExecuteStrategy) {
        MNN_DEBUG_PRINT("Strategy: %s\t%d\n", iter.first.c_str(), iter.second)
//        AUTOTIME;
        if (iter.first == "compute" || iter.first == "recompute") {
            if (iter.second >= mExecutions.size()) {
                continue;
            }
            max_computed = std::max(iter.second, max_computed);
            computeIthOp(iter.second, false, false, {}, true);
        } else {
            auto& cmd = mCmdBuffer.command[iter.second];
            for(auto t: cmd.outputs) {
                if (dynamic_type == 0) {
                    mBackend->onReleaseBuffer(t, Backend::DYNAMIC);
                } else if (dynamic_type == 1) {
                    mBackend->onFreeBufferToOS(t);
                } else if (dynamic_type == 2) {
                    mBackend->onFreeBufferHybrid(t);
                } else {
                    MNN_ASSERT(false)
                }
                allocatedTensor.erase(t);
            }
        }
    }

    mBackend->onExecuteEnd();
    mBackupBackend->onExecuteEnd();
//    mBackend->onClearBuffer();
//    mBackupBackend->onClearBuffer();

    return NO_ERROR;
}


ErrorCode Executor::ComputeCache::computeViaStrategy() {
    if (mComputeTarget == "capuchin") {
        mBackend->onClearBuffer();
        mBackupBackend->onClearBuffer();
    }
    MNN_DEBUG_PRINT("call %s: %lu - %lu\n", __FUNCTION__, mExecutions.size(), mExecuteStrategy.size())
    mBackend->onExecuteBegin();
    mBackupBackend->onExecuteBegin();
#ifndef ALLOCATE_CACHE_ID_RUNTIME
    for(int i=0; i<mCmdBuffer.command.size(); i++) {
        for(auto output: mCmdBuffer.command[i].outputs) {
            output->setCacheID(mUniqueCacheID++);
            tensorFromOp[output->cacheID()] = i;
        }
    }
#endif
    int max_computed = -1;
    for (auto iter: mExecuteStrategy) {
        MNN_DEBUG_PRINT("Strategy: %s\t%d\n", iter.first.c_str(), iter.second)
//        AUTOTIME;
        if (iter.first == "compute" || iter.first == "recompute") {
            if (iter.second >= mExecutions.size()) {
                continue;
            }
            max_computed = std::max(iter.second, max_computed);
            computeIthOp(iter.second, false, false, {}, true);
        } else {
            auto& cmd = mCmdBuffer.command[iter.second];
            for(auto t: cmd.outputs) {
                if (dynamic_type == 0) {
                    TensorUtils::getDescribe(t)->backend->onReleaseBuffer(t, Backend::DYNAMIC);
                } else if (dynamic_type == 1) {
                    TensorUtils::getDescribe(t)->backend->onFreeBufferToOS(t);
                } else if (dynamic_type == 2) {
                    TensorUtils::getDescribe(t)->backend->onFreeBufferHybrid(t);
                } else {
                    MNN_ASSERT(false)
                }
                allocatedTensor.erase(t);
            }
        }
    }
    while (max_computed < mExecutions.size()) {
        computeIthOp(max_computed++, false, false, {}, true);
    }
    mBackend->onExecuteEnd();
    mBackupBackend->onExecuteEnd();
    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::computeViaCheckpoint() {
    MNN_PRINT("call %s\n", __FUNCTION__ )
    // get checkpoint and skip-release node (batch normalization consts)
    char filename[100];
    sprintf(filename, "sublinear/%s/%s.sublinear.skip.txt", mModelname.c_str(), mModelname.c_str());
    MNN_DEBUG_PRINT("%s: filename = %s\n", __FUNCTION__ , filename)
    std::ifstream ifs(filename, std::ios::in);
    int a;
    std::set<int> skip_release;
    while (ifs >> a) {
        skip_release.insert(a);
    }
    ifs.close();

    sprintf(filename, "sublinear/%s/%s.sublinear.checkpoint.txt", mModelname.c_str(), mModelname.c_str());
    MNN_DEBUG_PRINT("%s: filename = %s\n", __FUNCTION__ , filename)
    ifs.open(filename, std::ios::in);
    std::vector<int> checkpointOps;
    while (ifs >> a) {
        checkpointOps.push_back(a);
    }
    ifs.close();

    int fp_thres = -1;
    if (mModelname == "MobilenetV2" || mModelname == "MobilenetV2_CL") {
        fp_thres = 1750;
    } else if (mModelname == "MobilenetV1") {
        fp_thres = 917;
    } else if (mModelname == "Squeezenet" || mModelname == "Squeezenet_CL") {
        fp_thres = 866;
    } else if (mModelname == "Googlenet"){
        fp_thres = 1849;
    } else if (mModelname == "SqueezenetNoBN") {
        fp_thres = 120;
    } else if (mModelname == "MobilenetV1NoBN") {
        fp_thres = 121;
    } else if (mModelname == "MobilenetV2NoBN") {
        fp_thres = 227;
    } else if (mModelname == "Resnet50"){
        fp_thres = 1835;
    } else if (mModelname == "Resnet50NoBN") {
        fp_thres = 298;
    }
//    std::vector<int> computeSequence = getComputeSequence("compute_list.txt");
//    for (int i = 0; i < computeSequence.size(); ++i) {
//        computeSequence[i]=i;
//    }
//    if (mCmdBuffer.command.size() != computeSequence.size()) {
//        MNN_DEBUG_PRINT("mExecutions.size() = %d, return compute directly\n", mCmdBuffer.command.size());
//        return computeDirectly();
//    }
//    std::vector<int> checkpointOps = {};
    //mobilenetV2: <=14
//    std::vector<int> checkpointOps = {66, 120, 174, 259, 283, 504, 885};  // ours
//    std::vector<int> checkpointOps = {40, 53, 70, 94, 99, 107, 120, 122, 148, 176, 259, 283, 296, 339, 504, 541, 749, 853, 888, 1076, 1126, 1324, 1397, 1436};  // CTQ
//40, 53, 70, 94, 99, 120, 148, 176, 259, 283, 296, 339, 504, 541, 749, 853, 888, 1076, 1126, 1324, 1397, 1436
    //alexnet: <=14
//    std::vector<int> checkpointOps = {19, 25, 33};  // ours
//    std::vector<int> checkpointOps = {19, 22, 25, 33, 39, 47, 56}; // ctq
    //squeezenet: >101
//    std::vector<int> checkpointOps = {4, 6, 17, 29, 40, 52, 73, 88};  // ours
//    std::vector<int> checkpointOps = {4, 6, 17, 32, 40, 55, 73, 88, 119};  // ctq
//    googlenet: >16 && <=16
//    std::vector<int> checkpointOps = {18, 22, };  // ours
//    std::vector<int> checkpointOps = {18, 21, 47, 128, 217};  // ctq
    if (checkpointOps.empty() || fp_thres == -1) {
        MNN_PRINT("%s: return to computeDirectly\n", __FUNCTION__ )
        return computeDirectly();
    }
    if (checkpointOps[0] != -1) {
        checkpointOps.insert(checkpointOps.begin(), -1);
    }
    if (checkpointOps[checkpointOps.size()-1] != fp_thres) {
        checkpointOps.push_back(fp_thres);
    }
    int currentCheckpointIdx = 0;
    mBackend->onExecuteBegin();
    mBackupBackend->onExecuteBegin();
    mBackend->onClearBuffer();
    mBackupBackend->onClearBuffer();
#ifndef ALLOCATE_CACHE_ID_RUNTIME
    for(int i=0; i<mCmdBuffer.command.size(); i++) {
        for(auto output: mCmdBuffer.command[i].outputs) {
            output->setCacheID(mUniqueCacheID++);
            tensorFromOp[output->cacheID()] = i;
        }
    }
#endif
    MNN_ASSERT(mExecutions.size() == mCmdBuffer.command.size());
    for (int i=0; i < mCmdBuffer.command.size(); ++i) {
        // Execution是否set了对于cmd的input和output tensors是没有影响的
        // execution仅仅定义了计算的逻辑，但是对于中间的依赖关系是没有影响的

        // 先保证每个input都是valid的，通过bfs找出来需要重新计算的op
        MNN_DEBUG_PRINT("check if need recompute for cmd[%d]\n", i)
        int needRecompute = -1;
        auto& cmd = mCmdBuffer.command[i];
        auto op = cmd.op;
        if (!cmd.buffer.empty()) {
            op = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
        }
        for (auto v = 0; v<cmd.inputs.size() && needRecompute == -1; ++v) {
            if (!SizeComputer::opNeedContent(op->type(), v)) {
                continue;
            }
            auto des = TensorUtils::getDescribe(cmd.inputs[v]);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
                MNN_DEBUG_PRINT("cmd[%d].input[%d] from cmd[%d], needRecompute = %d\n",
                                i, v, tensorFromOp[cmd.inputs[v]->cacheID()], int(opNeedRecompute[tensorFromOp[cmd.inputs[v]->cacheID()]]))
                if (des->useCount && opNeedRecompute[tensorFromOp[cmd.inputs[v]->cacheID()]]) {
                    needRecompute = tensorFromOp[cmd.inputs[v]->cacheID()];
                    break;
                }
            }
            for (auto& s : des->regions) {
                auto subDes = TensorUtils::getDescribe(s.origin);
                if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                    MNN_DEBUG_PRINT("cmd[%d].input[%d].region from cmd[%d], needRecompute = %d\n",
                              i, v, tensorFromOp[s.origin->cacheID()], int(opNeedRecompute[tensorFromOp[s.origin->cacheID()]]))
                    if (subDes->useCount && opNeedRecompute[tensorFromOp[s.origin->cacheID()]]) {
                        needRecompute = tensorFromOp[s.origin->cacheID()];
                        break;
                    }
                }
            }
        }
        if (needRecompute != -1) {
            // need to recompute
            MNN_DEBUG_PRINT("cmd[%d] need to recompute cmd[%d] first\n", i, needRecompute);
            int startRecomputeCheckpointIndex = std::lower_bound(checkpointOps.begin(), checkpointOps.end(), needRecompute) - checkpointOps.begin() - 1;
            MNN_DEBUG_PRINT("update useCount in [cmd[%d], cmd[%d]) due to recompute\n",
                            checkpointOps[startRecomputeCheckpointIndex] + 1, checkpointOps[startRecomputeCheckpointIndex + 1])
            for (int k = checkpointOps[startRecomputeCheckpointIndex] + 1; k < checkpointOps[startRecomputeCheckpointIndex + 1]; ++k) {
                auto& cmd_k = mCmdBuffer.command[k];
                auto op_k = cmd_k.op;
                if (!cmd_k.buffer.empty()) {
                    op_k = flatbuffers::GetMutableRoot<Op>(cmd_k.buffer.data());
                }
                for (auto v = 0; v<cmd_k.inputs.size(); ++v) {
                    if (!SizeComputer::opNeedContent(op->type(), v)) {
                        continue;
                    }
                    auto des = TensorUtils::getDescribe(cmd_k.inputs[v]);
                    if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
                        des->useCount+=1;
                        continue;
                    }
                    for (auto& s : des->regions) {
                        auto subDes = TensorUtils::getDescribe(s.origin);
                        if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                            subDes->useCount+=1;
                        }
                    }
                }
            }

            for (int k = checkpointOps[startRecomputeCheckpointIndex] + 1; k < checkpointOps[startRecomputeCheckpointIndex + 1]; ++k) {
                MNN_DEBUG_PRINT("start recompute execution[%d]\n", k)
                if (skip_release.find(k) == skip_release.end()) {
                    computeIthOp(k, false, true);
                }
                MNN_DEBUG_PRINT("\tfinish recompute execution[%d]\n", k);
            }
        }

        MNN_DEBUG_PRINT("start compute execution[%d]\n", i);
        computeIthOp(i, false, false, {checkpointOps[currentCheckpointIdx]});
        MNN_DEBUG_PRINT("\tfinish compute execution[%d]\n", i);
        // FP阶段，当遇到一个checkpoint的时候就把当前checkpoint和之前的checkpoint之间的临时feature-map都release
        // 等到之后需要的时候重新计算
        if (currentCheckpointIdx + 1 < checkpointOps.size() && i == checkpointOps[currentCheckpointIdx + 1]) {
            MNN_DEBUG_PRINT("reach a checkpoint[%d] = %d, release previous output\n", currentCheckpointIdx+1, checkpointOps[currentCheckpointIdx+1]);
            for (int idx = checkpointOps[currentCheckpointIdx] + 1; idx < checkpointOps[currentCheckpointIdx + 1]; ++idx) {
//                if (idx <= 16) {
//                    continue;
//                }
                if (skip_release.find(idx) != skip_release.end()) {
                    continue;
                }
                MNN_DEBUG_PRINT("try release cmd[%d]\n", idx);
                auto& cmd_idx = mCmdBuffer.command[idx];
                for (auto& output : cmd_idx.outputs) {
                    auto des = TensorUtils::getDescribe(output);
                    if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) { // MNN_PRINT("\tmatch with memory type\n");
                        if (des->usage == Tensor::InsideDescribe::NORMAL) { // MNN_PRINT("\tmatch with usage\n");
                            if (nullptr != des->backend) { // MNN_PRINT("\tmatch with backend\n");
                                if (des->useCount > 0) { // MNN_PRINT("\tmatch with use count\n");
                                    // 当前op的output是自己产生的，因此不需要判断opNeedRecompute这个标记，直接release就行，然后标记
                                    // swapout(output);
                                    if(dynamic_type == 0) {
                                        des->backend->onReleaseBuffer(output, Backend::DYNAMIC);
                                    } else if (dynamic_type == 1) {
                                        des->backend->onFreeBufferToOS(output);
                                    } else if (dynamic_type == 2) {
                                        des->backend->onFreeBufferHybrid(output);
                                    } else {
                                        MNN_ASSERT(false)
                                    }
                                    opNeedRecompute[idx] = true;
                                    allocatedTensor.erase(output);
                                    MNN_DEBUG_PRINT("\tsuccessfully release output of cmd[%d]\n", idx);
                                }
                            }
                        }
                    }
                }
            }
            MNN_DEBUG_PRINT("finish release temp tensors of [execution[%d] and execution[%d])\n", checkpointOps[currentCheckpointIdx], checkpointOps[currentCheckpointIdx+1]);
            currentCheckpointIdx++;
        }
    }
    mBackend->onExecuteEnd();
    mBackupBackend->onExecuteEnd();
    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::computeIthOp(int i, bool profile, bool recompute, std::vector<int> skipReleaseOpID, bool viaStrategy, bool enableSwap) {
#ifdef PROFILE_COST_IN_LOG
    AUTOTIME;
#endif
    // auto exe = ExecutorScope::Current();
    // std::ifstream ifs("/sys/class/thermal/thermal_zone0/temp", std::ios::in);
    // if (!ifs) {
    //     MNN_PRINT("failed to open temperature file.\n");
    //     MNN_ASSERT(false);
    // }
    // while (ifs) {
    //     float temp;
    //     ifs >> temp;
    //     if (ifs) {
    //         exe->mTemp.push_back(temp);
    //     }
    // }
    // ifs.close();

    auto& cmd = mCmdBuffer.command[i];
    auto op = cmd.op;
    bool origin = true;
    ErrorCode code;
    if (!cmd.buffer.empty()) {
        origin = false;
        op = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    }
#if defined(PROFILE_EXECUTION_IN_LOG) || defined(PROFILE_COST_IN_LOG) || defined(DEBUG_EXECUTION_DETAIL)
    MNN_PRINT("current Op is %dth:%d:%s\n", i, op->type(), EnumNameOpType(op->type()));
#endif

#ifdef MNN_EXPR_ENABLE_PROFILER
    Timer autoTime;
#endif
    // set execution
    if (mExecutions[i] == nullptr) {
//        mExecutions[i] = nullptr;
        bool cacheed = false;
        if (!mCacheExes.empty() && origin) {
            auto iter = mCacheExes.find(op);
            if (iter != mCacheExes.end()) {
                mExecutions[i] = iter->second;
                cacheed = true;
            }
        }
        if (nullptr == mExecutions[i]) {
            mExecutions[i].reset(mBackend->onCreate(cmd.inputs, cmd.outputs, op));
            if (nullptr == mExecutions[i]) {
                mExecutions[i].reset(mBackupBackend->onCreate(cmd.inputs, cmd.outputs, op));
            }
            if (nullptr == mExecutions[i]) {
                return NOT_SUPPORT;
            }
        }
        // Check if need wrap
        bool needWrap = false;
        auto bn = mExecutions[i]->backend();
        auto iterType = bn->type();
        for (int v = 0; v < cmd.inputs.size(); ++v) {
            if (!SizeComputer::opNeedContent(op->type(), v)) {
                continue;
            }
            auto inpDes = TensorUtils::getDescribe(cmd.inputs[v]);
            if (inpDes->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                for (auto &reg : inpDes->regions) {
                    auto orgDes = TensorUtils::getDescribe(reg.origin);
                    auto tensorBn = orgDes->backend;
                    auto type = MNN_FORWARD_CPU;
                    if (nullptr != tensorBn) {
                        type = tensorBn->type();
                    }
                    if (iterType != type) {
                        needWrap = true;
                        break;
                    }
                }
            } else {
                auto tensorBn = inpDes->backend;
                auto type = MNN_FORWARD_CPU;
                if (nullptr != tensorBn) {
                    type = tensorBn->type();
                }
                if (iterType != type) {
                    needWrap = true;
                    break;
                }
            }
            if (needWrap) {
                break;
            }
        }
        if (needWrap && (!cacheed)) {
            mExecutions[i] = std::make_shared<WrapExecution>(mBackupBackend.get(), mExecutions[i], false);
        }
        if ((op->type() == OpType_Convolution && cmd.inputs.size() == 1)) {
            // TODO: Support Other op's cache
            mCacheExes.insert(std::make_pair(op, mExecutions[i]));
        }
    }

    if (enableSwap) {
        for (auto v = 0; v<cmd.inputs.size(); ++v) {
            if (!SizeComputer::opNeedContent(op->type(), v)) {
                continue;
            }
            auto des = TensorUtils::getDescribe(cmd.inputs[v]);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
                MNN_DEBUG_PRINT("\tcmd[%d].input[%d] from cmd[%d], needSwapin = %d\n",
                                i, v, tensorFromOp[cmd.inputs[v]->cacheID()], int(featureSwapoutFlag[cmd.inputs[v]->cacheID()]))
                if (des->useCount && featureSwapoutFlag[cmd.inputs[v]->cacheID()]) {
                    des->backend->onAcquireBuffer(cmd.inputs[v], Backend::DYNAMIC);
                    swapin(cmd.inputs[v]);
                    featureSwapoutFlag[cmd.inputs[v]->cacheID()] = false;
                }
            }
            for (auto& s : des->regions) {
                auto subDes = TensorUtils::getDescribe(s.origin);
                if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                    MNN_DEBUG_PRINT("\tcmd[%d].input[%d].region from cmd[%d], needSwapin = %d\n",
                                    i, v, tensorFromOp[s.origin->cacheID()], int(featureSwapoutFlag[s.origin->cacheID()]))
                    if (subDes->useCount && featureSwapoutFlag[s.origin->cacheID()]) {
                        subDes->backend->onAcquireBuffer(s.origin, Backend::DYNAMIC);
                        swapin(s.origin);
                        featureSwapoutFlag[s.origin->cacheID()] = false;
                    }
                }
            }
        }
    }
    //allocate memory for output tensors
    auto bn = mExecutions[i]->backend();
    bn->changeBufferType(Backend::DYNAMIC_OUTPUT);
    for (auto t : cmd.outputs) {
#ifdef ALLOCATE_CACHE_ID_RUNTIME
        if (t->cacheID() == -1) {
            t->setCacheID(mUniqueCacheID++);
            tensorFromOp[t->cacheID()] = i;
        }
#endif
        auto des = TensorUtils::getDescribe(t);
        if (nullptr == des->backend || recompute || viaStrategy) {
            if (des->backend == nullptr) {
                TensorUtils::setLinearLayout(t);
                des->backend = bn;
            }
            auto rst = false;
            if (dynamic_type == 0) {
                rst = bn->onAcquireBuffer(t, Backend::DYNAMIC);
            } else if (dynamic_type == 1) {
                rst = bn->onRequireBufferFromOS(t);
            } else if (dynamic_type == 2) {
                rst = bn->onRequireBufferHybrid(t);
            } else {
                MNN_ASSERT(false)
            }
            if (!rst) {
                return OUT_OF_MEMORY;
            }
            MNN_DEBUG_PRINT("\tfinish allocate memory for cmd[%d].output\n", i)
        }
        allocatedTensor.insert(t);
    }

    // resize execution
    bn->changeBufferType(Backend::DYNAMIC_RESIZE);
//    MNN_PRINT("\tbegin resize cmd[%d]\n", i)
    code = mExecutions[i]->onResize(cmd.inputs, cmd.outputs);
//    MNN_PRINT("\tfinish resize cmd[%d]\n", i)
    if (NO_ERROR != code) {
        return code;
    }
    MNN_DEBUG_PRINT("\tfinish resize cmd[%d]\n", i)
//    if (mComputeHeuristically) {
//        for (auto t: cmd.outputs) {
//            if (dynamic_type == 0) {
//                bn->onReleaseBuffer(t, Backend::DYNAMIC);
//            } else if (dynamic_type == 1) {
//                bn->onFreeBufferToOS(t);
//            } else if (dynamic_type == 2) {
//                bn->onFreeBufferHybrid(t);
//            } else {
//                MNN_ASSERT(false)
//            }
//        }
//        return NO_ERROR;
//    }
    bn->changeBufferType(Backend::DYNAMIC_OTHER);
#ifdef MNN_EXPR_ENABLE_PROFILER
    float costTime = (float)autoTime.durationInUs() / (float)1000;
    ExecutorScope::Current()->addOpCostTime((int)op->type(), costTime);
#endif

#ifdef MNN_EXPR_ENABLE_PROFILER
    Timer autoTime;
#endif
    // execute
//    if (mExecutions.size() > 100) return NO_ERROR;
    if (mComputeTarget != "resize" && mComputeTarget != "profile") {
        // resize & profile_io不需要做计算
        // MNN_DEBUG_PRINT("\tbegin onExecute cmd[%d]\n", i)
        code = mExecutions[i]->onExecute(cmd.inputs, cmd.outputs);
        if (NO_ERROR != code) {
#ifdef MNN_EXPRESS_ERROR_REPORT
        auto op = cmd.buffer.empty() ? cmd.op : flatbuffers::GetRoot<Op>(cmd.buffer.data());
        MNN_ERROR("Error to compute for %s, \n", EnumNameOpType(op->type()));
#endif
            mBackend->onExecuteEnd();
            return code;
        }
        MNN_DEBUG_PRINT("\t%s: finish onExecute\n", __FUNCTION__ );
        if (enableSwap) {
            size_t current_size = mBackend->usedSize();
            MNN_DEBUG_PRINT("\tmBackend->usedSize() = %lu\n", current_size)
            if (typeid(mBackend) != typeid(mBackupBackend)) {
                current_size += mBackupBackend->usedSize();
                MNN_DEBUG_PRINT("\tafter add mBackupBackend->usedSize() = %lu\n", current_size)
            }
            while (current_size > budget << 20) {
                MNN_DEBUG_PRINT("\ttrigger swap out due to %lu > %lu\n", current_size, budget << 20)
                bool swapFlag = false;
                for (auto tid: featureMap) {
                    auto t = mCmdBuffer.command[tid].outputs[0];
                    if (allocatedTensor.find(t) != allocatedTensor.end() && !featureSwapoutFlag[tid]) {
                        swapout(t);
                        TensorUtils::getDescribe(t)->backend->onReleaseBuffer(t, Backend::DYNAMIC);
                        featureSwapoutFlag[tid] = true;
                        swapFlag = true;
                        break;
                    }
                }
                if (!swapFlag) {
                    break;
                }
                current_size = mBackend->usedSize();
                if (typeid(mBackend) != typeid(mBackupBackend)) {
                    current_size += mBackupBackend->usedSize();
                }
            }
        }
    }
//    mExecutions[i]->onResizeEnd();

    if (viaStrategy) {
        return NO_ERROR;
    }
#ifdef PROFILE_EXECUTION_IN_LOG
    // profile inputs outputs temps
    MNN_PRINT("\tinputs: [");
#endif
    for (auto v = 0; v<cmd.inputs.size(); ++v) {
        if (!SizeComputer::opNeedContent(op->type(), v)) {
            continue;
        }
        auto des = TensorUtils::getDescribe(cmd.inputs[v]);
        if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
#ifdef PROFILE_EXECUTION_IN_LOG
            MNN_PRINT("(%d %d), ", cmd.inputs[v]->cacheID(), cmd.inputs[v]->size());
#endif
            if (profile) {
                opInputs[i].insert(cmd.inputs[v]->cacheID());
            }
        }
        for (auto& s : des->regions) {
            auto subDes = TensorUtils::getDescribe(s.origin);
            if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
#ifdef PROFILE_EXECUTION_IN_LOG
                MNN_PRINT("(%d %d), ", s.origin->cacheID(), s.origin->size());
#endif
                if (profile) {
                    opInputs[i].insert(s.origin->cacheID());
                }
            }
        }
    }

#ifdef PROFILE_EXECUTION_IN_LOG
    MNN_PRINT("]\n\toutputs: [");
    for (auto & output : cmd.outputs) {
        MNN_PRINT("(%d %d), ", output->cacheID(), output->size());
    }
    MNN_PRINT("]\n\ttemporary: [");
    for (auto & output : cmd.outputs) {
        auto des = TensorUtils::getDescribe(output);
        if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
            MNN_PRINT("(%d %d), ", output->cacheID(), output->size());
            continue;
        }
    }
    MNN_PRINT("]\n\trelease: [");
#endif
    // release memory for no-usable tensors
    for (auto v = 0; v<cmd.inputs.size(); ++v) {
        if (!SizeComputer::opNeedContent(op->type(), v)) {
            continue;
        }
        auto t = cmd.inputs[v];
        auto des = TensorUtils::getDescribe(t);
        if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
            if (des->usage == Tensor::InsideDescribe::NORMAL) {
                des->useCount-=1;
//                MNN_DEBUG_PRINT("\t%s: cmd[%d].input[%d].useCount -- to %d\n", __FUNCTION__, i, v, des->useCount);
                if (nullptr != des->backend) {
                    if (0 == des->useCount &&
                            (skipReleaseOpID.empty() ||
                                std::find(skipReleaseOpID.begin(), skipReleaseOpID.end(), tensorFromOp[t->cacheID()]) != skipReleaseOpID.end())) {
                        MNN_DEBUG_PRINT("\t%s: release no-usable cmd[%d].input[%d] generated by cmd[%d]\n",
                               __FUNCTION__ , i, v, tensorFromOp[t->cacheID()]);
                        if (dynamic_type == 0) {
                            des->backend->onReleaseBuffer(t, Backend::DYNAMIC);
                        } else if (dynamic_type == 1) {
                            des->backend->onFreeBufferToOS(t);
                        } else if (dynamic_type == 2) {
                            des->backend->onFreeBufferHybrid(t);
                        } else {
                            MNN_ASSERT(false)
                        }
                        allocatedTensor.erase(t);
#ifdef PROFILE_EXECUTION_IN_LOG
                        MNN_PRINT("(%d %d), ", t->cacheID(), t->size());
#endif
                    }
                }
            }
        }
        int regidx = 0;
        for (auto& s : des->regions) {
            auto subDes = TensorUtils::getDescribe(s.origin);
            MNN_ASSERT(subDes->regions.empty());
            if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                subDes->useCount-=1;
//                MNN_DEBUG_PRINT("\t%s: cmd[%d].input[%d].region[%d].useCount -- to %d\n", __FUNCTION__, i, v, regidx, subDes->useCount);
                if (nullptr != subDes->backend) {
                    if (0 == subDes->useCount &&
                            (skipReleaseOpID.empty() ||
                            std::find(skipReleaseOpID.begin(), skipReleaseOpID.end(), tensorFromOp[tensorFromOp[s.origin->cacheID()]]) != skipReleaseOpID.end())) {
                        MNN_DEBUG_PRINT("\t%s: release no-usable cmd[%d].input[%d].region[%d] generated by cmd[%d]\n",
                                  __FUNCTION__ , i, v, regidx, tensorFromOp[s.origin->cacheID()]);
                        if (dynamic_type == 0) {
                            subDes->backend->onReleaseBuffer(s.origin, Backend::DYNAMIC);
                        } else if (dynamic_type == 1) {
                            subDes->backend->onFreeBufferToOS(s.origin);
                        } else if (dynamic_type == 2) {
                            subDes->backend->onFreeBufferHybrid(s.origin);
                        } else {
                            MNN_ASSERT(false)
                        }
                        allocatedTensor.erase(s.origin);
#ifdef PROFILE_EXECUTION_IN_LOG
                        MNN_PRINT("(%d %d), ", s.origin->cacheID(), s.origin->size());
#endif
                    }
                }
            }
            regidx++;
        }
    }
    MNN_DEBUG_PRINT("\t%s: finish release memory for no-usable tensors\n", __FUNCTION__ );
#ifdef PROFILE_EXECUTION_IN_LOG
    MNN_PRINT("]\n");
#endif

#ifdef MNN_EXPR_ENABLE_PROFILER
    float costTime = (float)autoTime.durationInUs() / (float)1000;
auto op = iter.op;
if (!iter.buffer.empty()) {
    op = flatbuffers::GetMutableRoot<Op>(iter.buffer.data());
}
ExecutorScope::Current()->addOpCostTime((int)op->type(), costTime);
#endif
    if(!profile) {
        opNeedRecompute[i] = false;
    }
    MNN_DEBUG_PRINT("\t%s: finish & return\n", __FUNCTION__ );
#ifdef PROFILE_COST_IN_LOG
    MNN_MEMORY_PROFILE("%s:%d", __FUNCTION__, i)
    MNN_PRINT("\t");
#endif
    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::resize() {
    if (!mShapeDirty) {
        return NO_ERROR;
    }
    for (auto& c : mInputInside) {
        if (c->mInfoDirty) {
            return CALL_BACK_STOP;
        }
    }
    if (!mInputs.empty()) {
        MNN_PRINT("%s: mInputCache.size() = %lu......shit\n", __FUNCTION__ , mInputs.size());
    }
    for (auto c : mInputs) {
        auto code = c->resize();
        MNN_PRINT("finish mInput.resize()\n");
        if (NO_ERROR != code) {
            return code;
        }
    }
    mShapeDirty = false;
    /** Encoder Begin */
    {
#ifdef MNN_EXPR_ENABLE_PROFILER
        {
Timer autoTime;
#endif
        mCmdBuffer.command.clear();
        mCmdBuffer.extras.clear();
        // 每个cache的backend都是重新new的，并不是通过shared_ptr
        // 但是bufferAllocater是共享的啊……所以为什么药在resize的时候clearbuffer？

        // 实际上每次训练都是把要计算的var压缩到了一个cache里面了
        // 所以这里的clear实际上没有用，因为每个batch只会有一个cache
        // 所以在每个batch结束的时候那些tensor都被删掉了，内存已经被返回给backend
        // 这里的allocated dynamic pool就是null的
        mExecutions.clear();
        mContext.clear();
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
ExecutorScope::Current()->addOpCostTime((int)OpType_While, costTime);
}
#endif
        CommandBuffer buffer;
        for (int unitIndex = 0; unitIndex < mUnits.size(); ++unitIndex) {
            auto& iter = *mUnits[unitIndex];
            auto inside = iter.inside.lock();
            if (nullptr == inside || inside->mInfoDirty) {
                mShapeDirty = true;
                continue;
            }
            // Check zero shape
            bool zeroShape = false;
            for (int i=0; i<iter.outputs.size(); ++i) {
                TensorUtils::copyShape(inside->mOutputTensors[i], iter.outputs[i], true);
                auto t = iter.outputs[i];
                // FIXME: Find better way to may compability for old model
                /**
                 For Convolution of 2D / 3D Tensor(Dense / 1D Convolution)
                 Because of old code, we will acces dim[2] / dim[3] to get width and height
                 Set the lenght to 1 for compability
                 */
                for (int v=t->dimensions(); v<4; ++v) {
                    t->setLength(v, 1);
                }
                iter.outputs[i]->buffer().type = inside->mOutputTensors[i]->buffer().type;
                auto des = TensorUtils::getDescribe(iter.outputs[i]);
                if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
                    des->backend = nullptr;
                }
                des->regions.clear();
                for (int v=0; v<t->dimensions(); ++v) {
                    if (t->length(v) == 0) {
                        zeroShape = true;
                        break;
                    }
                    if (t->length(v) < 0) {
                        return INPUT_DATA_ERROR;
                    }
                }
            }
            if (zeroShape) {
                // FIXME: for multi output and one tensor zero shape should support
                continue;
            }
#ifdef MNN_EXPR_ENABLE_PROFILER
            {
    Timer autoTime;
#endif
            auto geo = GeometryComputer::search(iter.op->type());
            geo->compute(iter.op, iter.inputs, iter.outputs, mContext, buffer);
#ifdef MNN_EXPR_ENABLE_PROFILER
            float costTime = (float)autoTime.durationInUs() / (float)1000;
        ExecutorScope::Current()->addOpCostTime((int)iter.op->type(), costTime);
    }
#endif
        }
#ifdef MNN_EXPR_ENABLE_PROFILER
        {
Timer autoTime;
#endif
        GeometryComputerUtils::makeRaster(buffer, mCmdBuffer, mContext);
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
ExecutorScope::Current()->addOpCostTime((int)OpType_If, costTime);
}
#endif
    }
    for (int k=0; k<mCmdBuffer.command.size(); ++k) {
        auto& cmd = mCmdBuffer.command[k];
        auto op = cmd.op;
        if (!cmd.buffer.empty()) {
            op = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
        }
        for (auto v = 0; v<cmd.inputs.size(); ++v) {
            if (!SizeComputer::opNeedContent(op->type(), v)) {
                continue;
            }
            auto des = TensorUtils::getDescribe(cmd.inputs[v]);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
                des->useCount+=1;
//                MNN_PRINT("%s: cmd[%d].input[%d].useCount ++ to %d\n", __FUNCTION__, k, v, des->useCount);
                continue;;
            }
            int regidx = 0;
            for (auto& s : des->regions) {
                auto subDes = TensorUtils::getDescribe(s.origin);
                if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                    subDes->useCount+=1;
//                    MNN_PRINT("%s: cmd[%d].input[%d].region[%d].useCount ++ to %d\n", __FUNCTION__, k, v, regidx++, subDes->useCount);
                }
            }
        }
    }
    /** Encoder End */

    /** Prepare Begin */
    if (!zeroInputs()) {
        MNN_PRINT("not zero inputs: %lu\tself.execution.size() = %lu\n", mInputs.size(), mCmdBuffer.command.size());
//        MNN_ASSERT(0)
        mBackend->onResizeBegin();
        mExecutions.resize(mCmdBuffer.command.size());
        for (int k=0; k<mCmdBuffer.command.size(); ++k) {
            auto& cmd = mCmdBuffer.command[k];
            auto op = cmd.op;
            bool origin = true;
            if (!cmd.buffer.empty()) {
                origin = false;
                op = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
            }
#ifdef MNN_EXPR_ENABLE_PROFILER
            Timer autoTime;
#endif
            mExecutions[k] = nullptr;
            bool cacheed = false;
            if (!mCacheExes.empty() && origin) {
                auto iter = mCacheExes.find(op);
                if (iter != mCacheExes.end()) {
                    mExecutions[k] = iter->second;
                    cacheed = true;
                }
            }
            if (nullptr == mExecutions[k]) {
                mExecutions[k].reset(mBackend->onCreate(cmd.inputs, cmd.outputs, op));
                if (nullptr == mExecutions[k]) {
                    mExecutions[k].reset(mBackupBackend->onCreate(cmd.inputs, cmd.outputs, op));
                }
                if (nullptr == mExecutions[k]) {
                    return NOT_SUPPORT;
                }
            }
            // Check if need wrap
            bool needWrap = false;
            auto bn = mExecutions[k]->backend();
            auto iterType = bn->type();
            for (int i=0; i<cmd.inputs.size(); ++i) {
                if (!SizeComputer::opNeedContent(op->type(), i)) {
                    continue;
                }
                auto inpDes = TensorUtils::getDescribe(cmd.inputs[i]);
                if (inpDes->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                    for (auto& reg : inpDes->regions) {
                        auto orgDes = TensorUtils::getDescribe(reg.origin);
                        auto tensorBn = orgDes->backend;
                        auto type = MNN_FORWARD_CPU;
                        if (nullptr != tensorBn) {
                            type = tensorBn->type();
                        }
                        if (iterType != type) {
                            needWrap = true;
                            break;
                        }
                    }
                } else {
                    auto tensorBn = inpDes->backend;
                    auto type = MNN_FORWARD_CPU;
                    if (nullptr != tensorBn) {
                        type = tensorBn->type();
                    }
                    if (iterType != type) {
                        needWrap = true;
                        break;
                    }
                }
                if (needWrap) {
                    break;
                }
            }
            if (needWrap && (!cacheed)) {
                mExecutions[k].reset(new WrapExecution(mBackupBackend.get(), mExecutions[k], false));
            }
            if ((op->type() == OpType_Convolution && cmd.inputs.size() == 1)) {
                // TODO: Support Other op's cache
                mCacheExes.insert(std::make_pair(op, mExecutions[k]));
            }
            for (auto t : cmd.outputs) {
                auto des = TensorUtils::getDescribe(t);
                if (nullptr == des->backend) {
                    TensorUtils::setLinearLayout(t);
                    auto res = bn->onAcquireBuffer(t, Backend::DYNAMIC);
                    des->backend = bn;
                    if (!res) {
                        return OUT_OF_MEMORY;
                    }
                }
            }
            auto code= mExecutions[k]->onResize(cmd.inputs, cmd.outputs);
            if (NO_ERROR != code) {
                return code;
            }
            for (auto v = 0; v<cmd.inputs.size(); ++v) {
                if (!SizeComputer::opNeedContent(op->type(), v)) {
                    continue;
                }
                auto t = cmd.inputs[v];
                auto des = TensorUtils::getDescribe(t);
                if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
                    if (des->usage == Tensor::InsideDescribe::NORMAL) {
                        des->useCount-=1;
                        if (0 == des->useCount && nullptr != des->backend) {
                            des->backend->onReleaseBuffer(t, Backend::DYNAMIC);
                        }
                    }
                }
                for (auto& s : des->regions) {
                    auto subDes = TensorUtils::getDescribe(s.origin);
                    MNN_ASSERT(subDes->regions.empty());
                    if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                        subDes->useCount-=1;
                        if (0 == subDes->useCount && nullptr != subDes->backend) {
                            subDes->backend->onReleaseBuffer(s.origin, Backend::DYNAMIC);
                        }
                    }
                }
            }
#ifdef MNN_EXPR_ENABLE_PROFILER
            float costTime = (float)autoTime.durationInUs() / (float)1000;
    ExecutorScope::Current()->addOpCostTime((int)op->type(), costTime);
#endif
        }
        mBackend->onResizeEnd();
    }
    /** Prepare End */

    mContentDirty = true;
    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::swapout(const Tensor *tensor) {
    MNN_DEBUG_PRINT("\t%s tensor[%d]\n", __FUNCTION__, tensor->cacheID())
    char fn[32];
    sprintf(fn, "swap/%d.mnn.tensor", tensor->cacheID());
    FILE* f = fopen(fn, "wb");
    int numwrite = -1;
    if(f != nullptr) {
        numwrite = fwrite(tensor->host<void>(), sizeof(char), tensor->size(), f);
//        MNN_PRINT("%d\n", numwrite);
    }
    if (numwrite != tensor->size()){
        return SWAP_OUT_ERROR;
    }
//    MNN_PRINT("swapout %d bytes of tensor:%d \n", tensor->size(), tensor->ID());
    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::swapin(const Tensor *tensor) {
    MNN_DEBUG_PRINT("\t%s tensor[%d]\n", __FUNCTION__, tensor->cacheID())
    char fn[32];
    sprintf(fn, "swap/%d.mnn.tensor", tensor->cacheID());
    FILE* f = fopen(fn, "rb");
    int numread = -1;
    if (f != nullptr) {
        numread = fread(tensor->host<void>(), sizeof(char), tensor->size(), f);
    }
    if (numread != tensor->size()){
        return SWAP_IN_ERROR;
    }
//    MNN_PRINT("swapin %d bytes of tensor:%d \n", tensor->size(), tensor->ID());
    return NO_ERROR;
}

ErrorCode Executor::ComputeCache::setExecutionStrategy(std::string model, int batch, int bgt) {
    mModelname = model;
    mBatchsize = batch;
    char filename[100];
    if (mComputeTarget == "ours" || mComputeTarget == "adaptive") {
        sprintf(filename, "heuristic/execution/%s/%s.%d.%d.execution.txt", model.c_str(), model.c_str(), batch, bgt);
    } else if (mComputeTarget == "capuchin") {
        // TODO: sprintf(filename, "capuchin/%s/%s.%d.%d.capuchin.txt", model.c_str(), model.c_str(), batch, bgt);
        sprintf(filename, "capuchin/%s/%s.%d.capuchin.txt", model.c_str(), model.c_str(), batch);
    }
    MNN_DEBUG_PRINT("%s: filename = %s\n", __FUNCTION__ , filename)
    std::ifstream ifs(filename, std::ios::in);
    std::string s;
    int a;
    while (ifs >> s >> a) {
        mExecuteStrategy.push_back(std::make_pair(s, a));
    }
    ifs.close();
    mComputeHeuristically = !mExecuteStrategy.empty();
    MNN_DEBUG_PRINT("mExecuteStrategy.size = %lu\n", mExecuteStrategy.size())

    return NO_ERROR;
}

void Executor::ComputeCache::config(std::string model, int batch) {
    mModelname = model;
    mBatchsize = batch;
}

void Executor::ComputeCache::setMethodAndTarget(std::string method, std::string target) {
    if (method == "direct" || method == "sublinear" || method == "strategy" || method == "swap" || method == "adaptive") {
        mComputeMethod = method;
    }
    if (target == "mnn" || target == "sublinear" || target == "ours" || target == "capuchin" || target == "vdnn" || target == "adaptive"
            || target == "profile" || target == "resize" || target == "cost") {
        mComputeTarget = target;
    }
}

ErrorCode Executor::ComputeCache::profileExecution() {
    MNN_PRINT("call %s\n", __FUNCTION__ );
    if (mShapeDirty) { // default true
        auto code = resize();
        if (zeroInputs()) {
            mExecutions.resize(mCmdBuffer.command.size());
        }
        if (NO_ERROR != code) {
            return code;
        }
    }
    /*在MNN训练的设定里面这两个for都没有什么实际的意义T^T*/
    for (auto& c : mInputInside) {
        if (c->mContentDirty) {
            // InputType = VARP::INPUT
            return CALL_BACK_STOP;
        }
    }
    for (auto c : mInputs) {
        auto code = c->compute();
        if (NO_ERROR != code) {
            return code;
        }
    }
//    MNN_PRINT("cache.minputs have %d content available directly and mExecution.size = %lu\n", cnt, mExecutions.size());
    if (mCmdBuffer.command.size()==1) {
        return NO_ERROR;
    }
    {
        // resize profile infos
        opInputs.resize(mExecutions.size());
        opLevel.resize(mExecutions.size());
        int outputsSize = 0;
        for (auto& cmd : mCmdBuffer.command) {
            outputsSize += cmd.outputs.size();
        }
        tensorFromOp.resize(outputsSize);
    }
    mBackend->onExecuteBegin();
    mBackupBackend->onExecuteBegin();
    MNN_PRINT("%s: mExecutions.size() = %lu\n", __FUNCTION__, mCmdBuffer.command.size());
#ifndef ALLOCATE_CACHE_ID_RUNTIME
    for(int i=0; i<mCmdBuffer.command.size(); ++i) {
        for(auto output: mCmdBuffer.command[i].outputs) {
            output->setCacheID(mUniqueCacheID++);
            tensorFromOp[output->cacheID()] = i;
        }
    }
#endif
    MNN_PRINT("%s: begin profile\n", __FUNCTION__ );
    MNN_ASSERT(mExecutions.size() == mCmdBuffer.command.size());
    for (int i=0; i<mCmdBuffer.command.size(); ++i) {
        ErrorCode code = computeIthOp(i, true);
        if(code != NO_ERROR) {
            opInputs.clear();
            opLevel.clear();
            tensorFromOp.clear();
            return code;
        }
    }
    mBackend->onExecuteEnd();
    mBackupBackend->onExecuteEnd();
    mContentDirty = false;
    getValidCheckpointLevel();
    MNN_PRINT("%s: finish & return\n", __FUNCTION__ );
    return NO_ERROR;

}

void Executor::ComputeCache::getValidCheckpointLevel() {
    // build input dependency opGraph
    /*std::vector<std::set<int>> opGraph, reversedOpGraph;
    opGraph.resize(opInputs.size());
    reversedOpGraph.resize(opInputs.size());

    std::vector<int> indegree;
    indegree.resize(opInputs.size());
    std::set<int> validCkpt;

    std::queue<int> que;
    // build graph
    for (int i = 0; i < opInputs.size(); ++i) {
        indegree[i] = 0;
        for (auto t: opInputs[i]) {
            int fromOp = tensorFromOp[t];
            opGraph[fromOp].insert(i);
            reversedOpGraph[i].insert(fromOp);
            indegree[i]++;
        }
    }
    MNN_PRINT("%s: finish build graph\n", __FUNCTION__ );
    // getValidCheckpointLevel via toposort
    for (int i = 0; i < indegree.size(); ++i) {
        if (indegree[i] == 0) {
            que.push(i);
            opLevel[i] = 0;
        }
    }
    while (!que.empty()) {
        int cur = que.front();
        que.pop();
        for (auto op: opGraph[cur]) {
            // cur -> op
            indegree[op]--;
            if(indegree[op] == 0) {
                que.push(op);
                opLevel[op] = opLevel[cur]+1;
            }
        }
    }
//    for (int i = 0; i < opLevel.size(); ++i) {
//        MNN_PRINT("%s: opLevel[%d] = %d\n", __FUNCTION__ , i, opLevel[i]);
//    }
//    MNN_PRINT("%s: getValidCheckpointLevel\n", __FUNCTION__ );
    // check valid ckpt level
    int threshold = opInputs.size() / 3;
    for (int i = 0; i < threshold; ++i) {
        MNN_PRINT("%s: opLevel[%d] = %d\n", __FUNCTION__ , i, opLevel[i]);
    }
    for (int i = 0; i < reversedOpGraph.size(); ++i) {
        MNN_PRINT("%s: reversedOpGraph[%d] = {", __FUNCTION__ , i);
        for (auto fromop: reversedOpGraph[i]) {
            MNN_PRINT("%d ", fromop);
        }
        MNN_PRINT("\n");
    }

    fpLevelList.resize(opLevel[threshold-1] + 1);
    for (int i = 0; i < threshold; ++i) {
        fpLevelList[opLevel[i]].push_back(i);
    }
    int idx = fpLevelList.size()-1;
    MNN_PRINT("%s: threshold = %d, init idx = %d\n", __FUNCTION__, threshold, idx);
    while (idx>=0) {
        std::vector<int> depends;
        for(auto op: fpLevelList[idx]) {
            for (auto fromOp: reversedOpGraph[op]) {
                if (opLevel[fromOp]) {
                    depends.push_back(opLevel[fromOp]);
                }
            }
        }
        if (depends.empty()) {
            idx--;
        } else {
            MNN_ASSERT(*std::max_element(depends.begin(), depends.end()) < idx)
            int minDepend = *std::min_element(depends.begin(), depends.end());
            if (minDepend+1 == idx) {
                validCkpt.insert(idx);
                validCkpt.insert(idx-1);
            }
            idx = minDepend;
        }
    }
    opGraph.clear();
    reversedOpGraph.clear();
//    opOutputs.clear();
    if(!validCkpt.empty()) {
        validCkptLevel.assign(validCkpt.begin(), validCkpt.end());
        profiledExecutionSizeWithValidCheckpoint = opInputs.size();
        MNN_PRINT("%s: finish check valid ckpt level\n", __FUNCTION__ );
        MNN_PRINT("%s: valid ckpt level has %d elements\n", __FUNCTION__, validCkptLevel.size());
//        for (auto ckptLevel: validCkptLevel) {
//            MNN_PRINT("fp level = %d is valid ckpt level with op: [", ckptLevel);
//            for (auto ckpt: fpLevelList[ckptLevel]) {
//                MNN_PRINT("%d ", ckpt);
//            }
//            MNN_PRINT("]\n");
//        }
        MNN_PRINT("%s: finish & return\n", __FUNCTION__ );
    }*/
    std::vector<std::set<int>> reversedOpGraph;
    reversedOpGraph.resize(opInputs.size());
    std::set<int> validCkpt;

    // build graph
    for (int i = 0; i < opInputs.size(); ++i) {
        for (auto t: opInputs[i]) {
            int fromOp = tensorFromOp[t];
            reversedOpGraph[i].insert(fromOp);
        }
    }
    int idx = opInputs.size()/3;
    for (int i = 0; i < reversedOpGraph.size(); ++i) {
        MNN_PRINT("%s: reversedOpGraph[%d] = {", __FUNCTION__ , i);
        for (auto fromop: reversedOpGraph[i]) {
            MNN_PRINT("%d ", fromop);
        }
        MNN_PRINT("\n");
    }
    while (idx >= 0) {
        if (reversedOpGraph[idx].empty()) {
            MNN_PRINT("reversedOpGraph[%d].empty()\n", idx);
            idx--;
        } else {
            int mindepends = opInputs.size();
            for (auto depends: reversedOpGraph[idx]) {
                if (depends < mindepends) {
                    mindepends = depends;
                }
            }
            MNN_PRINT("reversedOpGraph[%d].mindepends = %d\n", idx, mindepends);
            if (mindepends + 1 == idx) {
                // 这里没有idx：0->1->2->3->4   0->3    2->4
                // 此时1只依赖于0，但是因为0->2导致了
                validCkpt.insert(idx - 1);
            }
            idx = mindepends;
        }
    }
    if (!validCkpt.empty()) {
        profiledExecutionSizeWithValidCheckpoint=opInputs.size();
        validCkeckpoints.assign(validCkpt.begin(), validCkpt.end());
        std::sort(validCkeckpoints.begin(), validCkeckpoints.end());
        MNN_PRINT("%s: valid ckpt has %lu elements\n\t[", __FUNCTION__, validCkeckpoints.size());
        for (auto ckpt: validCkeckpoints) {
            MNN_PRINT("%d ", ckpt);
        }
        MNN_PRINT("]\n%s: finish & return\n", __FUNCTION__ );
    }
}

std::vector<int> Executor::ComputeCache::getComputeSequence(const char *filename) {
    if (filename == nullptr) {
        return {};
    }
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs) {
        return {};
    }
    std::vector<int>res;
    int a;
    while(ifs >> a) {
        res.push_back(a);
    }
    ifs.close();
    return std::move(res);
}

std::vector<int> Executor::ComputeCache::selectCheckpoint(int cnt) {
    /*int fp_thres = profiledExecutionSizeWithValidCheckpoint / 3;
    int interval = fp_thres / (cnt + 1);
    std::vector<int> ckpt_opid;
    std::vector<bool> visited;
    visited.resize(fpLevelList.size());
    int initCkptL = -1;
    for (int i = 0; i < fpLevelList.size(); ++i) {
        if (fpLevelList[i].size() == 1) {
            initCkptL = i;
            break;
        }
    }
    if(initCkptL == -1) {
        return ckpt_opid;
    }
    for (int i = 1; i <= cnt; ++i) {
        int validL = 0;
        for (auto ckptL: validCkeckpoints) {
            if(abs(ckptL - i * interval) < abs(validL- i * interval) && !visited[ckptL] && fpLevelList[ckptL].size() == 1) {
                // 这个if条件有bug，interval是根据op数量确定的，不能和ckptLevel直接比较
                validL = ckptL;
            }
        }
        ckpt_opid.push_back(fpLevelList[validL][0]);
    }
    return ckpt_opid;*/
    int fp_thres = profiledExecutionSizeWithValidCheckpoint / 3;
    int interval = fp_thres / (cnt + 1);
    std::vector<int> ckpt_opid;
    std::vector<bool> visited(opInputs.size());
    for (int i = 1; i <= cnt; ++i) {
        int valid = 0;
        for (auto ckpt: validCkeckpoints) {
            if(abs(ckpt - i * interval) < abs(valid - i * interval) && !visited[ckpt]) {
                valid = ckpt;
            }
        }
        ckpt_opid.push_back(valid);
    }
    return ckpt_opid;
}

std::vector<int> Executor::ComputeCache::getRecomputeOpList(int curOpID) {
    std::vector<int>recomputeList;
    std::queue<int> bfsQ;
    std::vector<bool> visit(opNeedRecompute.size());
    bfsQ.push(curOpID);

    while(!bfsQ.empty()) {
        int opid = bfsQ.front();
        bfsQ.pop();
        visit[opid] = true;
        recomputeList.push_back(opid);
        auto& cmd = mCmdBuffer.command[opid];
        auto op = cmd.op;
        if (!cmd.buffer.empty()) {
            op = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
        }
        for (auto v = 0; v<cmd.inputs.size(); ++v) {
            if (!SizeComputer::opNeedContent(op->type(), v)) {
                continue;
            }
            auto t = cmd.inputs[v];
            auto des = TensorUtils::getDescribe(t);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
                if (des->usage == Tensor::InsideDescribe::NORMAL) {
                    if (nullptr != des->backend) {
                        if (des->useCount && opNeedRecompute[tensorFromOp[t->cacheID()]] && !visit[tensorFromOp[t->cacheID()]]) {
                            bfsQ.push(tensorFromOp[t->cacheID()]);
                            visit[tensorFromOp[t->cacheID()]] = true;
                        }
                    }
                }
            }
            for (auto& s : des->regions) {
                auto subDes = TensorUtils::getDescribe(s.origin);
                MNN_ASSERT(subDes->regions.empty());
                if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                    if (nullptr != subDes->backend) {
                        if (subDes->useCount && opNeedRecompute[tensorFromOp[s.origin->cacheID()]] && !visit[tensorFromOp[s.origin->cacheID()]]) {
                            bfsQ.push(tensorFromOp[s.origin->cacheID()]);
                            visit[tensorFromOp[s.origin->cacheID()]] = true;
                        }
                    }
                }
            }
        }

    }
    std::sort(recomputeList.begin(), recomputeList.end());
    MNN_ASSERT(*recomputeList.rbegin() == curOpID)
    return std::move(recomputeList);
}

void Executor::ComputeCache::setBudgetAndProgress(size_t bgt, size_t bgt_adap, float adapProg) {
    budget = bgt;
    adaptiveBudget = bgt_adap;
    adaptiveProgress = adapProg;
}

static void _collectExecuteUnit(std::vector<std::shared_ptr<Executor::Unit>>& dest, EXPRP expr) {
    auto& inputs = expr->inputs();
    auto& req = expr->inside()->mReq.contentNeedContent;
    MNN_ASSERT(inputs.size() == req.size());
    
    for (int i=0; i<inputs.size(); ++i) {
        if (!req[i]) {
            continue;
        }
        auto inputExpr = inputs[i]->expr();
        auto unit = inputExpr.first->inside()->mUnit;
        if (nullptr == unit) {
            continue;
        }
        auto inputCache = inputExpr.first->inside()->mCache;
        if (nullptr != inputCache) {
            continue;
        }
        _collectExecuteUnit(dest, inputExpr.first);
    }
    auto unit = expr->inside()->mUnit;
    if (nullptr == unit) {
        return;
    }
    dest.emplace_back(std::move(unit));
    expr->inside()->mUnit = nullptr;
}

void Executor::_create(const std::vector<EXPRP>& outputs, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::set<std::shared_ptr<Expr::Inside>>&& inputNode, bool forceCPU) {
    std::vector<EXPRP> packed;
    for (auto expr : outputs) {
        auto cache = expr->inside()->mCache;
        if (nullptr != cache) {
            continue;
        }
        if (nullptr != expr->get()) {
            packed.emplace_back(expr);
            continue;
        }
    }
    if (packed.empty()) {
        return;
    }
    //MNN_PRINT("Create %p begin\n", packed[0].get());
    std::shared_ptr<Backend> cacheBn;
    std::shared_ptr<Backend> cacheBackupBn;
    if (forceCPU) {
        cacheBn.reset(mBackupRuntime.first->onCreate());
        cacheBackupBn = cacheBn;
    } else {
        cacheBn.reset(mRuntime.first->onCreate());
        cacheBackupBn.reset(mBackupRuntime.first->onCreate());
    }
    std::shared_ptr<ComputeCache> packedCache(new ComputeCache(cacheBn, cacheBackupBn));
    packedCache->config(mModelname, mBatchsize);
    if (mHeuristic) {  // 在计算整个model之前还有别的简单计算，这部分不需要通过 mHeuristic == false 过滤
        MNN_DEBUG_PRINT("%s: %s: mTarget=%s\n", __FILE_NAME__, __FUNCTION__, mTarget.c_str())
        if (mTarget == "profile" || mTarget == "resize" || mTarget == "cost") {
            packedCache->setMethodAndTarget("direct", mTarget);
            cacheBn->setHeuristicStrategy(mHeuristic, mModelname, mBatchsize, mBudgetMB, true);
            if (typeid(cacheBn) != typeid(cacheBackupBn)) {
                cacheBackupBn->setHeuristicStrategy(mHeuristic, mModelname, mBatchsize, mBudgetMB, true);
            }
        } else if (mTarget == "mnn") {
            packedCache->setMethodAndTarget("direct", mTarget);
        } else if (mTarget == "sublinear") {
            packedCache->setMethodAndTarget("sublinear", mTarget);
        } else if (mTarget == "ours") {
            cacheBn->setHeuristicStrategy(mHeuristic, mModelname, mBatchsize, mBudgetMB);
            if (typeid(cacheBn) != typeid(cacheBackupBn)) {
                cacheBackupBn->setHeuristicStrategy(mHeuristic, mModelname, mBatchsize, mBudgetMB);
            }
            packedCache->setMethodAndTarget("strategy", mTarget);
            packedCache->setExecutionStrategy(mModelname, mBatchsize, mBudgetMB);
        } else if (mTarget == "capuchin") {
            packedCache->setMethodAndTarget("strategy", mTarget);
            packedCache->setExecutionStrategy(mModelname, mBatchsize, mBudgetMB);
        } else if (mTarget == "vdnn") {
            packedCache->setMethodAndTarget("swap", mTarget);
        } else if (mTarget == "adaptive") {
            packedCache->setMethodAndTarget("adaptive", mTarget);
            packedCache->setExecutionStrategy(mModelname, mBatchsize, mBudgetMB);
            cacheBn->setHeuristicStrategy(mHeuristic, mModelname, mBatchsize, mBudgetMB);
            if (typeid(cacheBn) != typeid(cacheBackupBn)) {
                cacheBackupBn->setHeuristicStrategy(mHeuristic, mModelname, mBatchsize, mBudgetMB);
            }
        }
        packedCache->setBudgetAndProgress(mBudgetMB, mAdaptiveBudgetMB, mAdaptiveProgress);
    }

    packedCache->mInputs = std::move(inputCaches);
    packedCache->mInputInside = std::move(inputNode);
    for (auto expr : packed) {
        expr->inside()->mCacheOffset = (int)packedCache->mOutputs.size();
        MNN_ASSERT(expr->inside()->mUnit != nullptr);
        auto& originOutputs = expr->inside()->mUnit->outputs;
        for (auto t : originOutputs) {
            packedCache->mOutputs.emplace_back(t);
            TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
        }
    }
    for (auto expr : packed) {
        _collectExecuteUnit(packedCache->mUnits, expr);
//        MNN_PRINT("%s: mUnits.size() = %d\n", __FUNCTION__, packedCache->mUnits.size());
    }
//    MNN_ASSERT(0)
    for (auto expr : packed) {
        expr->inside()->mCache = packedCache;
    }
    //MNN_PRINT("Create %p End\n", packed[0].get());
}

void Executor::_visit(EXPRP expr, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::set<std::shared_ptr<Expr::Inside>>& inputNode) {
    auto& inputs = expr->inputs();
    auto& req = expr->inside()->mReq.contentNeedContent;
    MNN_ASSERT(inputs.size() == req.size());

    // Create Input's Unit / Cache
    for (int i=0; i<inputs.size(); ++i) {
        if (!req[i]) {
            continue;
        }
        //MNN_PRINT("Use %d\n", i);
        auto inputExpr = inputs[i]->expr();
        if (nullptr != inputExpr.first->inside()->mUnit) {
            continue;
        }
        auto inputCache = inputExpr.first->inside()->mCache;
        if (nullptr != inputCache) {
            MNN_PRINT("inputExpr.first->inside()->mCache != nullptr...mdzz\n");
            inputCaches.insert(inputCache);
            continue;
        }
        _visit(inputExpr.first, inputCaches, inputNode);
    }

    auto op = expr->get();
    if (nullptr == op) {
        return;
    }
    if (nullptr != expr->inside()->mUnit) {
        return;
    }
    std::shared_ptr<Unit> unitP(new Unit);
    Unit& unit = *unitP;
    unit.op = expr->get();
    unit.inside = std::weak_ptr<Expr::Inside>(expr->inside());
    unit.inputs.resize(inputs.size());
    unit.outputs.resize(expr->inside()->mOutputTensors.size());
    unit.outputContents.resize(unit.outputs.size());
    for (int i=0; i<unit.outputs.size(); ++i) {
        unit.outputContents[i].reset(new Tensor);
        unit.outputs[i] = unit.outputContents[i].get();
    }
    for (int i=0; i<inputs.size(); ++i) {
        auto inputExpr = inputs[i]->expr();
        unit.inputs[i] = inputExpr.first->inside()->mOutputTensors[inputExpr.second];
        if (!req[i]) {
            // The compute don't need it
            continue;
        }
        if (inputExpr.first->get() == nullptr) {
            if (inputExpr.first->inputType() == VARP::INPUT) {
                inputNode.insert(inputExpr.first->inside());
            }
            continue;
        }
        auto inputUnit = inputExpr.first->inside()->mUnit;
        if (nullptr != inputUnit) {
            unit.inputs[i] = inputUnit->outputs[inputExpr.second];
            continue;
        }
        MNN_ASSERT(nullptr != inputExpr.first->inside()->mCache);
        MNN_PRINT("inputExpr.first->inside()->mCache != nullptr...mdzz\n");
        inputCaches.insert(inputExpr.first->inside()->mCache);
        auto offset = inputExpr.second + inputExpr.first->inside()->mCacheOffset;
        unit.inputs[i] = inputExpr.first->inside()->mCache->mOutputs[offset];
    }
    MNN_ASSERT(expr->inside()->mUnit == nullptr);
    //MNN_PRINT("Create %p, %s\n", expr.get(), EnumNameOpType(expr->get()->type()));
    expr->inside()->mUnit = unitP;
}
void Executor::_makeCache(const std::vector<EXPRP>& expr, bool forceCPU) {
    std::set<std::shared_ptr<Executor::ComputeCache>> inputCaches;
    std::set<std::shared_ptr<Expr::Inside>> inputNode;
    for (auto e : expr) {
        _visit(e, inputCaches, inputNode);
    }
    _create(expr, std::move(inputCaches), std::move(inputNode), forceCPU);
}

void Executor::_checkTemp() {
    while (mTemp > mThresTemp) {
        sleep(5);
        std::cout << "Executor::_checkTemp: Current temperature is too high, waiting for cooling down..." << std::endl;
    }
}

void Executor::makeCache(const std::vector<EXPRP>& expr, bool forceCPU) {
    std::lock_guard<std::mutex> _l(mMutex);
    _checkTemp();
    //FUNC_PRINT(mCaches.size());
    _makeCache(expr, forceCPU);
}

void Executor::addOpCostTime(int op, float costTime) {
#ifdef MNN_EXPR_ENABLE_PROFILER
    auto opType = MNN::EnumNameOpType((OpType)op);
    if (nullptr == opType) {
        return;
    }
    mProfiler->add(opType, costTime);
#endif
}
void Executor::addOpCostTime(const std::string& type, float costTime) {
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->add(type, costTime);
#endif
}
void Executor::addOpFlops(const std::string& type, float flops) {
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->addFlops(type, flops);
#endif
}
void Executor::setHeuristicAlloc(bool flag) {
    mHeuristic = flag;
}
bool Executor::getHeuristicAllocFlag(){
    return mHeuristic;
}
void Executor::configExecution(std::string modelName, int batchsize, std::string target, size_t budgetMB, size_t adaptiveBudget, float adapProg) {
    mModelname = modelName;
    mBatchsize = batchsize;
    mTarget    = target;
    mBudgetMB = budgetMB;
    mAdaptiveBudgetMB = adaptiveBudget;
    mAdaptiveProgress = adapProg;
    MNN_DEBUG_PRINT("%s:          mModelname=%s,      mBatchsize=%d, mTarget=%s,      mBudgetMB=%lu, mAdaptiveBudgetMB=%lu, mAdaptiveProgress=%.1f\n",
                    __FUNCTION__, mModelname.c_str(), mBatchsize,    mTarget.c_str(), mBudgetMB,     mAdaptiveBudgetMB,     mAdaptiveProgress)
}


ErrorCode Executor::runCache(std::shared_ptr<ComputeCache> cache) {
    std::lock_guard<std::mutex> _l(mMutex);
    return cache->compute();
}
void Executor::resetProfile() {
    // mTemp.clear();
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->reset();
#endif
}
void Executor::dumpProfile() {
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->dump();
#endif
    // if (mTemp.empty()) {
    //     MNN_PRINT("No profile data to dump.\n");
    //     return;
    // }

    // for (auto& p : mTemp) {
    //     MNN_PRINT("%.1f ℃\n", p / 1000.0f);
    // }

    // mCounter = 0;

}

void Executor::profileCacheExecution(std::shared_ptr<ComputeCache> cache) {
    cache->profileExecution();
}

} // namespace Express
} // namespace MNN
