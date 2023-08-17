#!/bin/bash
cmake \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
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
#
adb push ./*.* /data/local/tmp/build_64/ > /dev/null
adb push tools/ /data/local/tmp/build_64/ > /dev/null
#adb push profile/ /data/local/tmp/build_64/ > /dev/null
#adb push resize/ /data/local/tmp/build_64/ > /dev/null
#adb push cost/ /data/local/tmp/build_64/ > /dev/null