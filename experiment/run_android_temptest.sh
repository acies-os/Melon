#!/bin/bash

ADB="/mnt/c/Users/capta/AppData/Local/Android/Sdk/platform-tools/adb.exe"

root_dir="/data/local/tmp/build_64"


cmake \
-DCMAKE_TOOLCHAIN_FILE=$HOME/android-ndk-r25c/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DANDROID_NATIVE_API_LEVEL=android-14  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_USE_LOGCAT=false \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DMNN_BUILD_TRAIN=ON \
-DMNN_BUILD_BENCHMARK=ON \
-DMNN_OPENCL=OFF \
-DPROFILE_EXECUTION_IN_LOG=OFF -DDEBUG_EXECUTION_DETAIL=OFF -DPROFILE_COST_IN_LOG=OFF \
../

make -j8

$ADB push ./*.* /data/local/tmp/build_64/ > /dev/null
$ADB push tools/ /data/local/tmp/build_64/ > /dev/null

$ADB shell "cd ${root_dir} && \
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
            chmod +x ./runTrainDemo.out && \
            ./runTrainDemo.out MemTimeProfile MobilenetV2 ../dataset/ ../dataset/train.txt ../dataset/ ../dataset/train.txt 40 40 mnn \
            > temptest_multithread.out"
