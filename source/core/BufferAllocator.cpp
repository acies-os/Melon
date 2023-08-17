//
//  BufferAllocator.cpp
//  MNN
//
//  Created by MNN on 2018/12/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/BufferAllocator.hpp"
#include "core/Macro.h"

//#define DUMP_USAGE
//#define MNN_DEBUG_MEMORY
namespace MNN {
class DefaultAllocator : public BufferAllocator::Allocator {
public:
    DefaultAllocator() {
        // Do nothing
    }
    virtual ~ DefaultAllocator() {
        // Do nothing
    }
    virtual std::pair<void*, size_t> onAlloc(size_t size) {
        MNN_DEBUG_PRINT("\tfail to reuse memory, require from OS with %lu byte\n", size)
        return std::make_pair(MNNMemoryAllocAlign(size, MNN_MEMORY_ALIGN_DEFAULT), 0);
    }
    virtual void onRelease(std::pair<void*, size_t> ptr) {
        MNN_ASSERT(ptr.second == 0);
        MNNMemoryFreeAlign(ptr.first);
    }
};
class RecurseAllocator : public BufferAllocator::Allocator {
public:
    RecurseAllocator(BufferAllocator* parent) {
        mParent = parent;
    }
    virtual ~ RecurseAllocator() {
        // Do nothing
    }
    virtual std::pair<void*, size_t> onAlloc(size_t size) override {
        return mParent->alloc(size);
    }
    virtual void onRelease(std::pair<void*, size_t> ptr) override {
        mParent->free(ptr);
    }
private:
    BufferAllocator* mParent;
};

std::shared_ptr<BufferAllocator::Allocator> BufferAllocator::Allocator::createDefault() {
    std::shared_ptr<BufferAllocator::Allocator> _res;
    _res.reset(new DefaultAllocator);
    return _res;
}
std::shared_ptr<BufferAllocator::Allocator> BufferAllocator::Allocator::createRecurse(BufferAllocator* parent) {
    std::shared_ptr<BufferAllocator::Allocator> _res;
    _res.reset(new RecurseAllocator(parent));
    return _res;
}

BufferAllocator::Node::~Node() {
    if (nullptr == parent) {
        outside->onRelease(pointer);
    }
}
std::pair<void*, size_t> BufferAllocator::alloc(size_t size, bool seperate) {
    MNN_DEBUG_PRINT("\t%s: call %s\n", mName.c_str(), __FUNCTION__ )
#ifdef DUMP_USAGE
    auto memoryUsed = size / 1024.0f / 1024.0f;
    MNN_PRINT("Alloc: %f\n", memoryUsed);
#endif
    std::pair<void*, size_t> pointer;
    // reuse if possible
    if (!seperate) {
        if (nullptr != mCurrentFreeList) {
            pointer = getFromFreeList(mCurrentFreeList, size, false);
        }
        if (nullptr != pointer.first) {
            mUsedSize += pointer.second;
            return pointer;
        }
        if (mName == "dynamic" && mCurrentFreeList == nullptr) {
            MNN_DEBUG_PRINT("\ttry get %lu bytes dynamic memory\n", UP_DIV(size, mAlign) * mAlign)
        }
        pointer = getFromFreeList(&mFreeList, size);
        if (nullptr != pointer.first) {
#ifdef DEBUG_EXECUTION_DETAIL
            if (mName == "dynamic") {
                debugUsage(__LINE__);
                MNN_DEBUG_PRINT("\tafter getFromFreeList, tot_size = %lu\n", mTotalSize)
            }
#endif
            mUsedSize += pointer.second;
//            MNN_DEBUG_PRINT("\t%s: successfully reused from mFreeList\n", mName.c_str())
            return pointer;
        }
//        debugUsage(__LINE__);
    }
    MNN_DEBUG_PRINT("\t%s: fail to get from free list, allocate otherwise\n", mName.c_str());

    // alloc otherwise
    pointer = mAllocator->onAlloc(size);
    if (nullptr == pointer.first) {
        return pointer;
    }
    mTotalSize += size;
    mUsedSize += size;

    // save node
    std::shared_ptr<Node> node(new Node);
    node->size         = size;
    node->pointer      = pointer;
    mUsedList[pointer] = node;
    node->outside      = mAllocator.get();

#ifdef DUMP_USAGE
    MNN_PRINT("mTotalSize: %f\n", mTotalSize / 1024.0f / 1024.0f);
#endif
#ifdef DEBUG_EXECUTION_DETAIL
    if (mName == "dynamic") {
        debugUsage(__LINE__);
        MNN_DEBUG_PRINT("\tafter alloc from os, tot_size = %lu\n", mTotalSize)
    }
#endif
    return pointer;
}

void BufferAllocator::setHeuristicStrategy(std::string model, int batch, int bgt, bool alignBottom, bool needAlloc) {
    release();
    char filename[100];
    if (alignBottom) {
        sprintf(filename, "heuristic/allocation/%s/%s.address.txt", model.c_str(), model.c_str());
    } else {
        sprintf(filename, "heuristic/allocation/%s/%s.%d.%d.address.txt", model.c_str(), model.c_str(), batch, bgt);
    }
    MNN_DEBUG_PRINT("%s: %s: filename = %s\n", mName.c_str(), __FUNCTION__ , filename)
    std::ifstream ifs(filename, std::ios::in);
    std::string s;
    size_t a;
    while (ifs >> s >> a) {
        if (s == "maxsize") {
            mHeuristicSize = a;
        } else {
            mHeuristicStrategy[s] = a;
        }
    }
    ifs.close();
//    if (!mHeuristicStrategy.empty()) {
//        mHeuristicStrategy.clear();
//        mHeuristicStrategy["0"] = 0;
//    }
    MNN_DEBUG_PRINT("%s: %s: maxsize = %lu\n", mName.c_str(), __FUNCTION__ , mHeuristicSize)
    if (mHeuristicSize && needAlloc) {
        auto heuristicPool = alloc(mHeuristicSize);
        mHeuristicPtr = heuristicPool.first;
        MNN_DEBUG_PRINT("%s: alloc mHeuristicPtr = %p\n", __FUNCTION__, mHeuristicPtr)
    }
}

std::pair<void*, size_t> BufferAllocator::allocHeuristically(std::string id, size_t size) {
    MNN_DEBUG_PRINT("\t%s: call %s\n",mName.c_str(), __FUNCTION__ )
//    debugUsage(__LINE__);
    if (mHeuristicStrategy.empty() || mDisableHeuristicWhileAdapting) {
        MNN_DEBUG_PRINT("\tmHeuristicStrategy is empty, return alloc()\n")
        return alloc(size, false);
    }
//    if (mHeuristicStrategy.find(id) == mHeuristicStrategy.end()) {
//        MNN_DEBUG_PRINT("\tid(%s) not in mHeuristicStrategy\n", id.c_str())
//        return alloc(size, false);
//        return std::make_pair(mHeuristicPtr, mHeuristicStrategy[id]);
//    }
#ifdef DEBUG_EXECUTION_DETAIL
    if (mName == "dynamic" && mCurrentFreeList == nullptr) {
        debugUsage(__LINE__);
        MNN_DEBUG_PRINT("\ttry get %lu bytes dynamic.heuristic memory at pos [%lu, %lu) %c %lu\n",
                        size, mHeuristicStrategy[id], size + mHeuristicStrategy[id], size + mHeuristicStrategy[id] < mHeuristicSize ? '<' : '>', mHeuristicSize)
    }
#endif
    mAllocatedSize[id] = size;
    return std::make_pair(mHeuristicPtr, std::min(mHeuristicStrategy[id], mHeuristicSize - size));
}

bool BufferAllocator::freeHeuristically(std::string id, std::pair<void*, size_t> pointer) {
    MNN_DEBUG_PRINT("\tcall %s\n", __FUNCTION__ )
    if (mHeuristicStrategy.empty() || mDisableHeuristicWhileAdapting) {
        return free(pointer);
    } else {
        MNN_DEBUG_PRINT("\ttry return %lu bytes to heuristic pool\n", mAllocatedSize[id])
        return true;
    }
}

std::pair<void*, size_t> BufferAllocator::allocFromOS(size_t size) {
    return std::make_pair(MNNMemoryAllocAlign(size, MNN_MEMORY_ALIGN_DEFAULT), 0);
}

bool BufferAllocator::freeToOS(std::pair<void*, size_t> ptr) {
    MNN_ASSERT(ptr.second == 0);
    MNNMemoryFreeAlign(ptr.first);
    return true;
}

void BufferAllocator::returnMemory(FREELIST* listP, std::shared_ptr<Node> node, bool permitMerge) {
//    MNN_PRINT("\tcall %s: %s\n", mName.c_str(), __FUNCTION__ )
    if (*listP == mFreeList && mName == "dynamic") {
        MNN_DEBUG_PRINT("\ttry return %lu bytes to freelist\n", node->size)
    }

    auto& list = *listP;
    list.insert(std::make_pair(node->size, node));
    // update parent use count
    if (nullptr != node->parent && permitMerge) {
        auto parent = node->parent;
        parent->useCount -= 1;

        // merge if all subnodes were freed
        auto needMerge = parent->useCount == 0;
        while (needMerge) {
            // collect all subnodes
            for (auto iter = list.begin(); iter != list.end();) {
                if (iter->second->parent == parent) {
                    iter = list.erase(iter);
                    continue;
                }
                iter++;
            }

            // do merge downside up
            list.insert(std::make_pair(parent->size, parent));
            needMerge = false;
            if (parent->parent != nullptr) {
                parent = parent->parent;
                parent->useCount -= 1;
                needMerge = parent->useCount == 0;
            }
        }
    }
}

bool BufferAllocator::free(std::pair<void*, size_t> pointer) {
    // get node
    MNN_DEBUG_PRINT("\t%s: call %s\n", mName.c_str(), __FUNCTION__ )
    auto x = mUsedList.find(pointer);
    if (x == mUsedList.end()) {
        MNN_ASSERT(false);
        return false;
    }
    // mark as reusable
    mUsedSize -= pointer.second;
    auto node = x->second;
    mUsedList.erase(x);
    if (nullptr != mCurrentFreeList) {
        returnMemory(mCurrentFreeList, node, false);
    } else {
        returnMemory(&mFreeList, node);
    }
//    debugUsage(__LINE__);
#ifdef DUMP_USAGE
    auto memoryUsed = x->second->size / 1024.0f / 1024.0f;
    MNN_PRINT("Free: %f\n", memoryUsed);
#endif
    return true;
}

void BufferAllocator::debugUsage(int line) const {
#ifdef DEBUG_EXECUTION_DETAIL
    size_t used_size = 0, free_size = 0;
    for(auto iter: mUsedList) {
        used_size += iter.second->size;
    }
    for (auto iter: mFreeList) {
        free_size += iter.first;
    }
    MNN_DEBUG_PRINT("\t%s: %s(%d): mUsedList.size = %lu with used_size = %lu, mFreeList.size = %lu with free_size = %lu, mUsedSize = %lu\n",
                    mName.c_str(), __FUNCTION__, line, mUsedList.size(), used_size, mFreeList.size(), free_size, mUsedSize)
#endif

}
size_t BufferAllocator::usedSize() const {
    MNN_DEBUG_PRINT("\tbefore return usedSize(), mUsedSize = %lu\n", mUsedSize)
    return mUsedSize;
}

void BufferAllocator::release(bool allRelease) {
    MNN_DEBUG_PRINT("%s: %s: %d\n", mName.c_str(), __FUNCTION__ , allRelease);
    MNN_ASSERT(mGroups.empty());
    debugUsage(__LINE__);
    if (allRelease) {
        mUsedList.clear();
        mFreeList.clear();
        mTotalSize = 0;
        mUsedSize = 0;
        return;
    }
    for (auto f : mFreeList) {
        if (f.second->parent == nullptr) {
            MNN_ASSERT(mTotalSize >= f.first);
//            ::free(f.second->pointer.first);
            mTotalSize -= f.first;
        }
    }
    mFreeList.clear();
    MNN_DEBUG_PRINT("%s: %s: return\n", mName.c_str(), __FUNCTION__ );
}

void BufferAllocator::barrierBegin() {
    MNN_ASSERT(mGroups.empty());
}

void BufferAllocator::barrierEnd() {
    for (auto& freeGroup : mGroups) {
        auto freeList = *freeGroup;
        for (auto& iter : freeList) {
            returnMemory(&mFreeList, iter.second);
        }
    }
    mGroups.clear();
}

void BufferAllocator::beginGroup() {
    std::shared_ptr<FREELIST> newFreeList(new FREELIST);
    mCurrentFreeList = newFreeList.get();
    mGroups.emplace_back(newFreeList);
}

void BufferAllocator::endGroup() {
    mCurrentFreeList = nullptr;
}

std::pair<void*, size_t> BufferAllocator::getFromFreeList(FREELIST* list, size_t size, bool permiteSplit) {
//    MNN_PRINT("\t%s: call %s\n", mName.c_str(), __FUNCTION__ )
#ifdef MNN_DEBUG_MEMORY
    return std::make_pair(nullptr, 0);
#endif

    // get node larger than size
#ifdef DEBUG_EXECUTION_DETAIL
    size_t tot_size = 0;
    for (auto iter: *list) {
        tot_size += iter.first;
    }
    MNN_DEBUG_PRINT("\t%s: free list.size = %lu, tot_size = %lu\n", mName.c_str(), list->size(), tot_size)
#endif
    auto x = list->lower_bound(size);
    if (x == list->end()) {
#ifdef DEBUG_EXECUTION_DETAIL
        if (list->size()) {
            auto max_size = std::max_element(
                        list->begin(), list->end(),
                        [](const std::pair<size_t, std::shared_ptr<Node>>& p1, const std::pair<size_t, std::shared_ptr<Node>>& p2){
                            return p1.first < p2.first;
                        }
                    )->first;
            MNN_DEBUG_PRINT("\t%s: free list is not empty with max_size = %lu, but fail to match size = %lu\n", mName.c_str(), max_size, size)
        }
#endif
        return std::make_pair(nullptr, 0);
    }
    MNN_DEBUG_PRINT("\t%s: match size %lu with %lu\n", mName.c_str(), size, x->first)

    // update parent use count
    auto pointer = x->second->pointer;
    if (permiteSplit && nullptr != x->second->parent) {
        x->second->parent->useCount += 1;
    }

    // uses up all aligned space
    auto sizeAlign = UP_DIV(size, mAlign) * mAlign;
    if (sizeAlign >= x->first || (!permiteSplit)) {
        mUsedList.insert(std::make_pair(pointer, x->second));
        list->erase(x);
        return pointer;
    }

    // split otherwise
    std::shared_ptr<Node> first(new Node);
    first->parent  = x->second;
    first->size    = sizeAlign;
    first->pointer = x->second->pointer;
    first->outside = mAllocator.get();
    mUsedList.insert(std::make_pair(pointer, first));
    x->second->useCount += 1;

    std::shared_ptr<Node> second(new Node);
    second->outside = mAllocator.get();
    second->parent  = x->second;
    second->size    = x->second->size - sizeAlign;
    second->pointer.first = x->second->pointer.first;
    second->pointer.second = x->second->pointer.second + sizeAlign;
    list->erase(x);
    list->insert(std::make_pair(second->size, second));
    return pointer;
}

std::vector<Tensor*> BufferAllocator::moveTensor2bottom(std::vector<Tensor *> tensorList, size_t bgt_new_mb) {
    std::sort(tensorList.begin(), tensorList.end(),
              [this](Tensor* a, Tensor* b) {
                  return mHeuristicStrategy[std::to_string(a->cacheID())] < mHeuristicStrategy[std::to_string(b->cacheID())];
              }
    );
    // TODO: realloc doen not work, just simulate
//    mHeuristicSize = bgt_new_mb * 1024 * 1024;
//    mHeuristicPtr = realloc(mHeuristicPtr, mHeuristicSize);
    MNN_DEBUG_PRINT("%s: realloc mHeuristicPtr = %p\n", __FUNCTION__, mHeuristicPtr)
    mTotalSize = mHeuristicSize;
    shrinkPointer = 0;
    for (auto t: tensorList) {
        if (mHeuristicStrategy[std::to_string(t->cacheID())] + t->size() < mHeuristicSize) {
//            MNN_DEBUG_PRINT("\t before memcpy\n")
            memcpy((uint8_t*)mHeuristicPtr + shrinkPointer, t->host<uint8_t>(), t->size());
//            MNN_DEBUG_PRINT("\t before adaptHost\n")
            t->adaptHost((uint8_t*)mHeuristicPtr + shrinkPointer);
//            MNN_DEBUG_PRINT("\t before shrinkPointer += t->size()\n")
            shrinkPointer += t->size();
//            MNN_DEBUG_PRINT("\t before final\n")
            tensorReversedAfterShrink.push_back(t);
        } else {
            break;
        }
    }
    MNN_DEBUG_PRINT("\t finish moving tensors to bottom\n")
    mUsedSize = shrinkPointer;
    mDisableHeuristicWhileAdapting = true;

    // 方便后续重新计算未在内存中的tensor，充分利用空间
    std::shared_ptr<Node> par(new Node);
    par->size = mHeuristicSize;
    par->outside = mAllocator.get();
    par->pointer = std::make_pair(mHeuristicPtr, mHeuristicSize);

    std::shared_ptr<Node> node(new Node);
    node->size = mHeuristicSize - shrinkPointer;
    node->pointer = std::make_pair((void*)((uint8_t*)mHeuristicPtr + shrinkPointer), mHeuristicSize - shrinkPointer);
    node->parent = par;
    mFreeList.insert(std::make_pair(mHeuristicSize - shrinkPointer, node));

    return tensorReversedAfterShrink;

}

bool BufferAllocator::adaptTensorToNewAddress(std::vector<Tensor*> tensors) {
    std::sort(tensors.begin(), tensors.end(),
              [this](Tensor* a, Tensor* b) {
                  return mHeuristicStrategy[std::to_string(a->cacheID())] < mHeuristicStrategy[std::to_string(b->cacheID())];
              }
    );
    MNN_DEBUG_PRINT("%s: moving recomputed tensors to bottom\n", __FUNCTION__)
    for(auto t: tensors) {
        if(std::find(tensorReversedAfterShrink.begin(), tensorReversedAfterShrink.end(), t) == tensorReversedAfterShrink.end()) {
            memcpy((uint8_t*)mHeuristicPtr + shrinkPointer, t->host<uint8_t>(), t->size());
            t->adaptHost((uint8_t*)mHeuristicPtr + shrinkPointer);
            shrinkPointer += t->size();
        }
    }
    MNN_DEBUG_PRINT("%s: re-locate all tensors according to new plan, \n", __FUNCTION__)
    // 按照新的策略重新排布
    for(auto iter = tensors.rbegin(); iter != tensors.rend(); iter++) {
        auto t = *iter;
        memcpy((uint8_t*)mHeuristicPtr + mHeuristicStrategy[std::to_string(t->cacheID())], t->host<uint8_t>(), t->size());
    }
    // remove nodes due to recompute for adaptiveness
    mFreeList.clear();
    mUsedList.clear();
    return true;
}
} // namespace MNN
