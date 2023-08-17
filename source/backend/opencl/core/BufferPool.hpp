//
//  BufferPool.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BufferPool_hpp
#define BufferPool_hpp

#include <map>
#include <memory>
#include <vector>
#include "core/NonCopyable.hpp"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"

namespace MNN {
namespace OpenCL {
class BufferPool : public NonCopyable {
public:
    BufferPool(cl::Context& context, cl_mem_flags flags) : mContext(context) {
        mFlag = flags;
        mUsedSize = 0;
        wasted = 0;
        freed = 0;
        mTotalSize = 0;
    }

    cl::Buffer* alloc(size_t size, bool seperate = false);
    void recycle(cl::Buffer* buffer, bool release = false);
    void clear();
    //全部分配的
    //当前真正使用的
    //有一部分因为不能拆分buffer导致的浪费
    //当前可用的
    // total_size = used + wasted + freed
    size_t mUsedSize, wasted, freed;

    struct Node {
        size_t size;
        std::shared_ptr<cl::Buffer> buffer;
    };

    void testAlloc(size_t msize);
    size_t totalSize() const {
        return mTotalSize;
    }
    size_t usedSize() const {
        return mUsedSize;
    }

private:
    std::map<cl::Buffer*, std::shared_ptr<Node>> mAllBuffer;
    std::multimap<size_t, std::shared_ptr<Node>> mFreeList;

    cl::Context& mContext;
    cl_mem_flags mFlag;
    size_t  mTotalSize;
};
class BufferPoolInt8 : public NonCopyable {
public:
    BufferPoolInt8(cl::Context& context, cl_mem_flags flags) : mContext(context) {
        mFlag = flags;
        mTotalSize = 0;
        mUsedSize  = 0;
    }

    cl::Buffer* alloc(int size, bool seperate = false);
    void recycle(cl::Buffer* buffer, bool release = false);
    void clear();
    size_t totalSize() const {
        return mTotalSize;
    }
    size_t usedSize() const {
        return mUsedSize;
    }

    struct Node {
        int size;
        std::shared_ptr<cl::Buffer> buffer;
    };

private:
    std::map<cl::Buffer*, std::shared_ptr<Node>> mAllBuffer;
    std::multimap<int, std::shared_ptr<Node>> mFreeList;

    cl::Context& mContext;
    cl_mem_flags mFlag;
    size_t mTotalSize, mUsedSize;
};
} // namespace OpenCL
} // namespace MNN

#endif /* BufferPool_hpp */
