//
//  BufferPool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/BufferPool.hpp"
namespace MNN {
namespace OpenCL {
cl::Buffer* BufferPool::alloc(size_t size, bool seperate) {
//    MNN_DEBUG_PRINT("\t%s: %d\t%d\n", __FILEPATH__, __FUNCTION__, __LINE__, size)
    if (!seperate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer.get();
            mFreeList.erase(iter);
            freed -= iter->first;
            mUsedSize += iter->first;
//            MNN_PRINT("\tafter reuse from freelist, total_size = %lu, free_size = %lu, reused_size = %d\n", total_alloc, freed, iter->first);
            MNN_DEBUG_PRINT("\t%s: reuse %lu bytes buffer\n", __FUNCTION__, iter->first)
            return buffer;
        }
    }
    std::shared_ptr<Node> node(new Node);
    node->size = size;
    MNN_DEBUG_PRINT("\t%s: try new %lu bytes buffer\n", __FUNCTION__, size)
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size));
    mTotalSize += size;
    mUsedSize += size;
//    MNN_PRINT("\tafter new Buffer, total_size = %lu, free_size = %lu\n", total_alloc, freed);
    mAllBuffer.insert(std::make_pair(node->buffer.get(), node));

    return node->buffer.get();
}

void BufferPool::recycle(cl::Buffer* buffer, bool release) {
    auto iter = mAllBuffer.find(buffer);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    mUsedSize -= iter->second->size;
    if (release) {
        MNN_DEBUG_PRINT("\t%s: release %lu bytes buffer\n", __FUNCTION__, iter->second->size)
        mTotalSize -= iter->second->size;
        mAllBuffer.erase(iter);
        return;
    }
    MNN_DEBUG_PRINT("\t%s: recycle %lu bytes buffer to freelist\n", __FUNCTION__, iter->second->size)
    mFreeList.insert(std::make_pair(iter->second->size, iter->second));
}

void BufferPool::clear() {
    mFreeList.clear();
    mAllBuffer.clear();
    freed = 0;
    mTotalSize = 0;
    mUsedSize = 0;
}

void BufferPool::testAlloc(size_t msize) {


    cl_platform_id *platformALL = nullptr;
    cl_uint ret_num_platforms;

    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices;

    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;

    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_int ret, err;

    const char *kernel_src_str = "\n" \
        "__kernel void addVec(__global int *dataSrc){"\
        "float16 aa; int idx = get_global_id(0);"\
        "if (idx<10){"\
        "	dataSrc[idx] += 10;"\
        "}}";

    //Step1:获取平台列表
    err = clGetPlatformIDs(0, nullptr, &ret_num_platforms);
    if (ret_num_platforms < 1) {
        MNN_PRINT("Error: Getting Platforms; err = %d,numPlatforms = %d !", err, ret_num_platforms);
    }
    MNN_PRINT("Num of Getting Platforms = %d!\n", ret_num_platforms);

    platformALL = (cl_platform_id *) alloca(sizeof(cl_platform_id) * ret_num_platforms);
    ret = clGetPlatformIDs(ret_num_platforms, platformALL, &ret_num_platforms);

    //Step2:获取指定设备，platformALL[0],platformALL[1]...
    //带有独显的PC,选择intel核显或独显
    ret = clGetDeviceIDs(platformALL[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    char nameDevice[64];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(nameDevice), &nameDevice, nullptr);
    MNN_PRINT("Device Name: %s\n", nameDevice);

    //Step3:创建上下文
    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
    if (ret < 0) {
        MNN_PRINT("clCreateContext Fail,ret =%d\n", ret);
        exit(1);
    }

    //Step4:创建命令队列
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret < 0) {
        MNN_PRINT("clCreateCommandQueue Fail,ret =%d\n", ret);
        exit(1);
    }

    //Step5:创建程序
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_src_str, nullptr, &ret);
    if (ret < 0) {
        MNN_PRINT("clCreateProgramWithSource Fail\n");
        exit(1);
    }
    //Step6:编译程序
    ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if (ret < 0) {
        MNN_PRINT("clBuildProgram Fail\n");
        exit(1);
    }

    //Step7:创建内核
    kernel = clCreateKernel(program, "addVec", &ret);
    if (ret < 0) {
        MNN_PRINT("clCreateKernel Fail\n");
        exit(1);
    }

    MNN_PRINT("GPU openCL init Finish\n");
    /*  fail to overlap Buffer with host_ptr via CL_MEM_USE_HOST_PTR flag

    cl_mem clDataSrcBuf;
//    int dataSrcHost[LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    //创建缓存对象
//#ifdef MEM_MAP
//    clDataSrcBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * LENGTH, nullptr, &err);
//#else
//    clDataSrcBuf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * LENGTH, dataSrcHost, &err);
//#endif
    int LENGTH = msize / sizeof(float);
    float *host_ptr = (float *) malloc(msize);
    float *res_ptr = (float *) malloc(msize);
    memset(res_ptr, 0, msize);
    for (int i = 0; i < LENGTH; i++) {
        host_ptr[i] = i;
    }
//    use Buffer
    clDataSrcBuf = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, msize, host_ptr, &err);
    err = clEnqueueReadBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, msize, res_ptr, 0, nullptr, nullptr);
    MNN_PRINT("the init buffer data is:\n");
    for (int i = 0; i < LENGTH; ++i) {
        MNN_PRINT("%.0lf, ", res_ptr[i]);
        if (i % 32 == 31) {
            MNN_PRINT("\n");
        }
    }
    MNN_PRINT("\n");
//        use Image
//        cl_image_desc desc;
//        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
//        desc.image_width = 32;
//        desc.image_height = 32;
//        desc.image_depth = 0;
//        desc.image_array_size = 0;
//        desc.image_row_pitch = 0;
//        desc.image_slice_pitch = 0;
//        desc.num_mip_levels = 0;
//        desc.num_samples = 0;
//        desc.buffer = nullptr;
//        cl_image_format format;
//        format.image_channel_order = CL_R;
//        format.image_channel_data_type = CL_FLOAT;
//        clDataSrcBuf = clCreateImage2D(context, CL_MEM_USE_HOST_PTR, &format, 32, 32, 0, host_ptr, &err);
    if (err < 0) {
//        CL_INVALID_HOST_PTR
        MNN_PRINT("clCreateBuffer imgSrcBuf Fail,err=%d\n", err);
        exit(1);
    }
    // 修改host_ptr，查看buffer
    for (int i = 0; i < LENGTH; i++) {
        host_ptr[i] += i;
    }
    // read image
//        size_t origin[3] = {0, 0, 0}; // Offset within the image to copy form
//        size_t region[3] = {32, 32, 1}; // Elements to per dimension
//        err = clEnqueueReadImage(command_queue, clDataSrcBuf, CL_TRUE,
//                                 origin, region, 0, 0,
//                                 res_ptr, 0, nullptr, nullptr);
    // read buffer
    memset(res_ptr, -1, msize);
    err = clEnqueueReadBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, msize, res_ptr, 0, nullptr, nullptr);
    MNN_PRINT("after modify host_ptr, the [read-buffer:host] data is expected to be the same:\n");
    for (int i = 0; i < LENGTH; ++i) {
        MNN_PRINT("[%.0lf:%.0lf], ", res_ptr[i], host_ptr[i]);
        if (i % 32 == 31) {
            MNN_PRINT("\n");
        }
        res_ptr[i] = 12345;
    }
    MNN_PRINT("\n");
    MNN_PRINT("after reset res_ptr, the res_ptr data is expected to be 12345:\n");
    for (int i = 0; i < LENGTH; ++i) {
        MNN_PRINT("%.0lf, ", res_ptr[i]);
        if (i % 32 == 31) {
            MNN_PRINT("\n");
        }
    }
    MNN_PRINT("\n");
    err = clEnqueueWriteBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, msize, res_ptr, 0, nullptr, nullptr);
    memset(res_ptr, 0, msize);
    MNN_PRINT("after write buffer and reset res_ptr, the [res_ptr:host_ptr] data is:\n");
    for (int i = 0; i < LENGTH; ++i) {
        MNN_PRINT("[%.0lf:%.0lf], ", res_ptr[i], host_ptr[i]);
        if (i % 32 == 31) {
            MNN_PRINT("\n");
        }
    }
    MNN_PRINT("\n");
    err = clEnqueueReadBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, msize, res_ptr, 0, nullptr, nullptr);
    MNN_PRINT("the buffer data is expected to be 12345:\n");
    for (int i = 0; i < LENGTH; ++i) {
        MNN_PRINT("%.0lf, ", res_ptr[i]);
        if (i % 32 == 31) {
            MNN_PRINT("\n");
        }
    }
    MNN_PRINT("\n");
    free(res_ptr);
    free(host_ptr);
    return;
//    err = clEnqueueReadImage(command_queue, clDataSrcBuf, CL_TRUE,
//                             origin, region, 0, 0,
//                             res_ptr, 0, nullptr, nullptr);
//        memset(res_ptr, 0, sizeof(int) * LENGTH);
//        err = clEnqueueReadImage(command_queue, clDataSrcBuf, CL_TRUE,
//                                 origin, region, 0, 0,
//                                 res_ptr, 0, nullptr, nullptr);
//        for (int i = 0; i < LENGTH; ++i) {
//            printf("%d:%d, ", host_ptr[i], res_ptr[i]);
//            if (i % 32 == 31) {
//                printf("\n");
//            }
//        }
//        printf("\n");
    return;
    MNN_PRINT("successfully test alloc buffer with size %lu\n", msize) */

    int LENGTH = msize / sizeof(float);
    void* srcptr = malloc(msize);
    cl_mem clDataSrcBuf = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, msize, srcptr, &err);
    float* host_ptr = (float *)malloc(msize);
    for(int i=0; i<LENGTH; i++) {
        host_ptr[i] = i;
    }
    err = clEnqueueWriteBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, msize, host_ptr, 0, nullptr, nullptr);
    MNN_PRINT("finish write src-buf\n")
    memset(host_ptr, -1, msize);
    cl_buffer_region region;
    region.origin = LENGTH / 4 * sizeof(float);
//        region.origin = 0;
    region.size = LENGTH / 2 * sizeof(float);
    cl_mem subBuf = clCreateSubBuffer(clDataSrcBuf, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if(err != CL_SUCCESS) {
        MNN_PRINT("Cannot set buffers: %d\t%d\n", err, CL_DEVICE_MEM_BASE_ADDR_ALIGN)
        exit(1);
    }
    err = clEnqueueReadBuffer(command_queue, subBuf, CL_TRUE, 0, region.size, host_ptr, 0, nullptr, nullptr);
    MNN_PRINT("the inited sub-buffer data is expected to be i:\n");
    for (int i = 0; i < region.size / sizeof(float); ++i) {
        MNN_PRINT("%.0lf, ", host_ptr[i]);
    }
    MNN_PRINT("\n");
    for(int i=0; i<LENGTH; i++) {
        host_ptr[i] = i * i;
    }
    err = clEnqueueWriteBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, msize, host_ptr, 0, nullptr, nullptr);
    memset(host_ptr, -1, msize);
    err = clEnqueueReadBuffer(command_queue, subBuf, CL_TRUE, 0, region.size, host_ptr, 0, nullptr, nullptr);
    MNN_PRINT("the after write the src-buffer, the sub-buffer data is expected to be i*i:\n");
    for (int i = 0; i < region.size / sizeof(float); ++i) {
        MNN_PRINT("%.0lf, ", host_ptr[i]);
    }
    MNN_PRINT("\n");
    memset(host_ptr, 0, msize);
    err = clEnqueueWriteBuffer(command_queue, subBuf, CL_TRUE, 0, region.size, host_ptr, 0, nullptr, nullptr);
    memset(host_ptr, -1, msize);
    err = clEnqueueReadBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, msize, host_ptr, 0, nullptr, nullptr);
    MNN_PRINT("after write the sub-buffer, the src-buffer data is hybrid (i*i or 0):\n");
    for (int i = 0; i < LENGTH; ++i) {
        MNN_PRINT("%.0lf, ", host_ptr[i]);
    }
    MNN_PRINT("\n");
}

cl::Buffer* BufferPoolInt8::alloc(int size, bool seperate) {
//    MNN_PRINT("%s: %s: %d\n", __FILEPATH__, __FUNCTION__, __LINE__)
    if (!seperate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer.get();
            mFreeList.erase(iter);
            mUsedSize += iter->first;
            return buffer;
        }
    }
    std::shared_ptr<Node> node(new Node);
    node->size = size;
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size));
    mTotalSize += size;
    mUsedSize += size;
    mAllBuffer.insert(std::make_pair(node->buffer.get(), node));

    return node->buffer.get();
}

void BufferPoolInt8::recycle(cl::Buffer* buffer, bool release) {
    auto iter = mAllBuffer.find(buffer);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    mUsedSize -= iter->second->size;
    if (release) {
        mTotalSize -= iter->second->size;
        mAllBuffer.erase(iter);
        return;
    }
    mFreeList.insert(std::make_pair(iter->second->size, iter->second));
}

void BufferPoolInt8::clear() {
    mFreeList.clear();
    mAllBuffer.clear();
    mTotalSize = 0;
    mUsedSize = 0;
}
} // namespace OpenCL
} // namespace MNN
