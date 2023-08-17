#!/bin/bash

models=("MobilenetV2" "Squeezenet" "MobilenetV1" "Resnet50")
root_dir="/data/local/tmp/build_64"
adb push profile/ $root_dir > /dev/null
adb push resize/ $root_dir > /dev/null
adb push cost/ $root_dir > /dev/null

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

for model in ${models[@]}; do
    adb shell "mkdir -p ${root_dir}/data/profiler/${model}"
    adb shell "mkdir -p ${root_dir}/data/heu_info/${model}"
    adb shell "mkdir -p ${root_dir}/heuristic/execution/${model}"
    adb shell "mkdir -p ${root_dir}/heuristic/allocation/${model}"
done

for budget in 8 6; do
    for model in ${models[@]}; do
        if [ $model == "MobilenetV1" ] && [ $budget -eq 8 ]; then
            batchs=(32 64 96 128)
        elif [ $model == "Resnet50" ] && [ $budget -eq 8 ]; then
            batchs=(32 64 96)
        elif [ $model == "MobilenetV2" ] && [ $budget -eq 8 ]; then
            batchs=(40 64 80 96 112 128 144 160 176 192 208)
        elif [ $model == "Squeezenet" ] && [ $budget -eq 8 ]; then
            batchs=(48 64 80 96 112 128 144 160 176 192)
        elif [ $model == "MobilenetV2" ] && [ $budget -eq 6 ]; then
            batchs=(40 48 64 80 96 112 128)
        elif [ $model == "Squeezenet" ] && [ $budget -eq 6 ]; then
            batchs=(48 64 80 96 112 128)
        else
            continue
        fi

        for batch in ${batchs[@]}; do
            echo ${model} ${batch} ${budget}
            adb shell "cd ${root_dir} && \
                        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
                        ./runTrainDemo.out GeneratePlan ${model} ${batch} ${budget}"
            adb shell "cd ${root_dir} && \
                        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
                        ./runTrainDemo.out GeneratePlan ${model} ${batch} ${budget} true"
        done
    done
done

adb pull /data/local/tmp/build_64/heuristic/ ./ > /dev/null