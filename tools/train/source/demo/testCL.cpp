//#include <ctime>
//#include <cstdio>
//#include <cstdlib>
//#include <sys/types.h>
//#include <cstring>
//#include <memory.h>
//#include <algorithm>
//
//#ifdef __APPLE__
//#include <OpenCL/cl.h>
//#else
//#include <CL/cl.h>
//#endif
//
//#ifdef WIN32
//#include <windows.h>
//#include <time.h>
//#include <direct.h>
//#include <io.h>
//#else
//#include <ctime>
//#include <unistd.h>
//#endif
//
//#include "DemoUnit.hpp"
//
//#define MAX_SOURCE_SIZE (0x100000)
//#define LENGTH       1024
//#define KERNEL_FUNC  "addVec"
//
//#define MEM_MAP    //映射内存对象
//
//class testCL : public DemoUnit {
//public:
//    virtual int run(int argc, const char *argv[]) override {
//#ifdef MEM_MAP
//        printf("fuck\n");
//#endif
//
//        cl_platform_id *platformALL = nullptr;
//        cl_uint ret_num_platforms;
//
//        cl_device_id device_id = nullptr;
//        cl_uint ret_num_devices;
//
//        cl_context context = nullptr;
//        cl_command_queue command_queue = nullptr;
//
//        cl_program program = nullptr;
//        cl_kernel kernel = nullptr;
//
//        cl_int ret, err;
//
//        const char *kernel_src_str = "\n" \
//        "__kernel void addVec(__global int *dataSrc){"\
//        "float16 aa; int idx = get_global_id(0);"\
//        "if (idx<10){"\
//        "	dataSrc[idx] += 10;"\
//        "}}";
//
//        //Step1:获取平台列表
//        err = clGetPlatformIDs(0, nullptr, &ret_num_platforms);
//        if (ret_num_platforms < 1) {
//            printf("Error: Getting Platforms; err = %d,numPlatforms = %d !", err, ret_num_platforms);
//        }
//        printf("Num of Getting Platforms = %d!\n", ret_num_platforms);
//
//        platformALL = (cl_platform_id *) alloca(sizeof(cl_platform_id) * ret_num_platforms);
//        ret = clGetPlatformIDs(ret_num_platforms, platformALL, &ret_num_platforms);
//
//        //Step2:获取指定设备，platformALL[0],platformALL[1]...
//        //带有独显的PC,选择intel核显或独显
//        ret = clGetDeviceIDs(platformALL[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
//
//        char nameDevice[64];
//        clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(nameDevice), &nameDevice, nullptr);
//        printf("Device Name: %s\n", nameDevice);
//
//        //Step3:创建上下文
//        context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
//        if (ret < 0) {
//            printf("clCreateContext Fail,ret =%d\n", ret);
//            exit(1);
//        }
//
//        //Step4:创建命令队列
//        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
//        if (ret < 0) {
//            printf("clCreateCommandQueue Fail,ret =%d\n", ret);
//            exit(1);
//        }
//
//        //Step5:创建程序
//        program = clCreateProgramWithSource(context, 1, (const char **) &kernel_src_str, nullptr, &ret);
//        if (ret < 0) {
//            perror("clCreateProgramWithSource Fail\n");
//            exit(1);
//        }
//        //Step6:编译程序
//        ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
//        if (ret < 0) {
//            perror("clBuildProgram Fail\n");
//            exit(1);
//        }
//
//        //Step7:创建内核
//        kernel = clCreateKernel(program, KERNEL_FUNC, &ret);
//        if (ret < 0) {
//            perror("clCreateKernel Fail\n");
//            exit(1);
//        }
//
//        printf("GPU openCL init Finish\n");
//
//        cl_mem clDataSrcBuf;
////    int dataSrcHost[LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//
//        //创建缓存对象
////#ifdef MEM_MAP
////    clDataSrcBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * LENGTH, nullptr, &err);
////#else
////    clDataSrcBuf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * LENGTH, dataSrcHost, &err);
////#endif
//        int *host_ptr = (int *) malloc(LENGTH * sizeof(int));
//        int *res_ptr = (int *) malloc(LENGTH * sizeof(int));
//        memset(res_ptr, 0, sizeof(int) * LENGTH);
//        for (int i = 0; i < 1024; i++) {
//            host_ptr[i] = i;
//        }
////    use Buffer
////    clDataSrcBuf = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, LENGTH * sizeof(int), host_ptr, &err);
////    use Image
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
//        if (err < 0) {
////        CL_INVALID_HOST_PTR
//            printf("clCreateBuffer imgSrcBuf Fail,err=%d\n", err);
//            exit(1);
//        }
//        for (int i = 0; i < 1024; i++) {
//            host_ptr[i] += i;
//        }
//        size_t origin[3] = {0, 0, 0}; // Offset within the image to copy form
//        size_t region[3] = {32, 32, 1}; // Elements to per dimension
//        err = clEnqueueReadImage(command_queue, clDataSrcBuf, CL_TRUE,
//                                 origin, region, 0, 0,
//                                 res_ptr, 0, nullptr, nullptr);
//        for (int i = 0; i < LENGTH; ++i) {
//            printf("%d, ", res_ptr[i]);
//            if (i % 32 == 31) {
//                printf("\n");
//            }
//            res_ptr[i] = 12345;
//        }
//        printf("\n");
//        err = clEnqueueWriteImage(command_queue, clDataSrcBuf, CL_TRUE,
//                                  origin, region, 0, 0,
//                                  res_ptr, 0, nullptr, nullptr);
////    memset(res_ptr, 0, sizeof(int) * LENGTH);
////    err = clEnqueueReadImage(command_queue, clDataSrcBuf, CL_TRUE,
////                             origin, region, 0, 0,
////                             res_ptr, 0, nullptr, nullptr);
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
//        return 0;
//
////#ifdef MEM_MAP
////    cl_int *bufferMap = (cl_int *) clEnqueueMapBuffer(command_queue, clDataSrcBuf, CL_TRUE, CL_MAP_WRITE,
////                                                      0, LENGTH * sizeof(cl_int), 0, nullptr, nullptr, nullptr);
////    memcpy(bufferMap, dataSrcHost, 10 * sizeof(int));
////#endif
//
//        //设置内核参数
//        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clDataSrcBuf);
//        if (err < 0) {
//            perror("clSetKernelArg imgDstBuf Fail\n");
//            exit(1);
//        }
//
//        printf("GPU openCL Create and set Buffter Finish\n");
//        cl_uint work_dim = 1;
//        size_t global_item_size = LENGTH;
//
//        err = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, nullptr, &global_item_size, nullptr, 0, nullptr, nullptr);
//        clFinish(command_queue);
//        if (err < 0) {
//            printf("err:%d\n", err);
//            perror("clEnqueueNDRangeKernel Fail\n");
//
//        }
//
////#ifndef MEM_MAP
////    err = clEnqueueReadBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, (4 * LENGTH), dataSrcHost, 0, nullptr, nullptr);
////    if (err < 0) {
////        printf("err:%d\n", err);
////        perror("Read buffer command Fail\n");
////
////    }
////#endif
//
//        //print result
//        for (int i = 0; i < LENGTH; i++) {
////#ifdef MEM_MAP
////        printf("dataSrcHost[%d]:%d\n", i, bufferMap[i]);
////#else
//            printf("dataSrcHost[%d]:%d\n", i, host_ptr[i]);
////#endif
//        }
//
////#ifdef MEM_MAP
////    err = clEnqueueUnmapMemObject(command_queue, clDataSrcBuf, bufferMap, 0, nullptr, nullptr);
////#endif
//
//        /* OpenCL Object Finalization */
//        ret = clReleaseKernel(kernel);
//        ret = clReleaseProgram(program);
//        ret = clReleaseCommandQueue(command_queue);
//        ret = clReleaseContext(context);
//
//        return 0;
//    }
//};
//
//DemoUnitSetRegister(testCL, "testCL");