//
//  ImagePool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/ImagePool.hpp"
namespace MNN {
namespace OpenCL {
cl::Image* ImagePool::alloc(int w, int h, cl_channel_type type, bool seperate) {
    if (!seperate) {
        int minWaste  = 0;
        auto findIter = mFreeList.end();
        for (auto iterP = mFreeList.begin(); iterP != mFreeList.end(); iterP++) {
            auto& iter = *iterP;
            if (iter->w >= w && iter->h >= h) {
                int waste = iter->w * iter->h - w * h;
                if (minWaste == 0 || waste < minWaste) {
                    findIter = iterP;
                    minWaste = waste;
                }
            }
        }
        if (findIter != mFreeList.end()) {
            auto image = (*findIter)->image.get();
            mFreeList.erase(findIter);
            mUsedSize += (*findIter)->w * (*findIter)->h * 4 * sizeof(float);
            MNN_DEBUG_PRINT("%s:%s: reuse Image with [w, h] = [%d, %d]\n", __FILEPATH__, __FUNCTION__, (*findIter)->w, (*findIter)->h)
            return image;
        }
    }
    std::shared_ptr<Node> node(new Node);
    node->w = w;
    node->h = h;
    node->image.reset(
        new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, type), w, h, 0, nullptr, nullptr));
    mTotalSize += w * h * 4 * sizeof(float);
    mUsedSize += w * h * 4 * sizeof(float);
    if (nullptr == node->image) {
        MNN_ERROR("All Image %d x %d error \n", w, h);
        return nullptr;
    }
    MNN_DEBUG_PRINT("%s:%s: alloc Image with [w, h] = [%d, %d]\n", __FILEPATH__, __FUNCTION__, w, h)
    mAllImage.insert(std::make_pair(node->image.get(), node));
    return node->image.get();
}

void ImagePool::recycle(cl::Image* image, bool release) {
    auto iter = mAllImage.find(image);
    if (iter == mAllImage.end()) {
        MNN_ERROR("recycle failed for not belong image\n");
        return;
    }
    mUsedSize -= iter->second->w * iter->second->h * 4 * sizeof(float);
    if (release) {
        MNN_DEBUG_PRINT("%s:%s: release Image with [w, h] = [%d, %d]\n", __FILEPATH__, __FUNCTION__, iter->second->w, iter->second->h)
        mTotalSize -= iter->second->w * iter->second->h * 4 * sizeof(float);
        mAllImage.erase(iter);
        return;
    }
    MNN_DEBUG_PRINT("%s:%s: recycle Image with [w, h] = [%d, %d]\n", __FILEPATH__, __FUNCTION__, iter->second->w, iter->second->h)
    mFreeList.push_back(iter->second);
}

void ImagePool::clear() {
    mTotalSize = 0;
    mUsedSize = 0;
    mFreeList.clear();
    mAllImage.clear();
}

cl::Image* ImagePoolInt8::alloc(int w, int h, bool seperate) {
    if (!seperate) {
        int minWaste  = 0;
        auto findIter = mFreeList.end();
        for (auto iterP = mFreeList.begin(); iterP != mFreeList.end(); iterP++) {
            auto& iter = *iterP;
            if (iter->w >= w && iter->h >= h) {
                int waste = iter->w * iter->h - w * h;
                if (minWaste == 0 || waste < minWaste) {
                    findIter = iterP;
                    minWaste = waste;
                }
            }
        }
        if (findIter != mFreeList.end()) {
            auto image = (*findIter)->image.get();
            mFreeList.erase(findIter);
            mUsedSize += (*findIter)->w * (*findIter)->h * 4 * sizeof(int8_t);
            return image;
        }
    }
    std::shared_ptr<Node> node(new Node);
    node->w = w;
    node->h = h;
    node->image.reset(
        new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, mType), w, h, 0, nullptr, nullptr));
    mTotalSize += w * h * 4 * sizeof(int8_t);
    mUsedSize += w * h * 4 * sizeof(int8_t);
    if (nullptr == node->image) {
        MNN_ERROR("All Image %d x %d error \n", w, h);
        return nullptr;
    }
    mAllImage.insert(std::make_pair(node->image.get(), node));
    return node->image.get();
}

void ImagePoolInt8::recycle(cl::Image* image, bool release) {
    auto iter = mAllImage.find(image);
    if (iter == mAllImage.end()) {
        MNN_ERROR("recycle failed for not belong image\n");
        return;
    }
    mUsedSize -= iter->second->w * iter->second->h * 4 * sizeof(int8_t);
    if (release) {
        mTotalSize -= iter->second->w * iter->second->h * 4 * sizeof(int8_t);
        mAllImage.erase(iter);
        return;
    }
    mFreeList.push_back(iter->second);
}

void ImagePoolInt8::clear() {
    mTotalSize = 0;
    mUsedSize = 0;
    mFreeList.clear();
    mAllImage.clear();
}
} // namespace OpenCL
} // namespace MNN
