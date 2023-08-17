#!/bin/bash

#models=("MobilenetV2" "Squeezenet" "MobilenetV1" "Resnet50")
root_dir="/data/local/tmp/build_64"
#models=("MobilenetV2NoBN" "SqueezenetNoBN" "MobilenetV1NoBN" "Resnet50NoBN")
models=("MobilenetV2" "Squeezenet" "MobilenetV1" "Resnet50")

for model in ${models[@]}; do
    adb shell "cd ${root_dir} && \
                mkdir -p heuristic/allocation/${model} && \
                mkdir -p heuristic/execution/${model} && \
                echo maxsize 2500000000 > heuristic/allocation/${model}/${model}.address.txt && \
                echo 0 0 >> heuristic/allocation/${model}/${model}.address.txt"
done
#exit

for target in "profile" "resize" "cost"; do
    if [[ $target == "profile" ]]; then
        build_opt="-DPROFILE_EXECUTION_IN_LOG=ON -DDEBUG_EXECUTION_DETAIL=OFF -DPROFILE_COST_IN_LOG=OFF"
    elif [[ $target == "resize" ]]; then
        build_opt="-DPROFILE_EXECUTION_IN_LOG=OFF -DDEBUG_EXECUTION_DETAIL=ON -DPROFILE_COST_IN_LOG=OFF"
    else
        build_opt="-DPROFILE_EXECUTION_IN_LOG=OFF -DDEBUG_EXECUTION_DETAIL=OFF -DPROFILE_COST_IN_LOG=ON"
    fi

    echo $build_opt

#    build
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
        ${build_opt} \
        ../
#    continue
    make -j8

    adb push ./*.* /data/local/tmp/build_64/ > /dev/null
    adb push tools/ /data/local/tmp/build_64/ > /dev/null
#    adb push heuristic/ /data/local/tmp/build_64/ > /dev/null

    for model in ${models[@]}; do
        output_dir="${target}/${model}"
        mkdir -p $output_dir

        if [[ $model == "MobilenetV1" ]]; then
            batchs=(32 64 96 128)
        elif [[ $model == "Resnet50" ]]; then
            batchs=(32 64 96)
        elif [[ $model == "MobilenetV2" ]]; then
            batchs=(40 48 64 80 96 112 128 144 160 176 192 208)
        elif [[ $model == "Squeezenet" ]]; then
            batchs=(48 64 80 96 112 128 144 160 176 192)
        fi

        for batch in ${batchs[@]}; do
            echo $model $batch $target
            adb shell "rm ${root_dir}/memory_profile.out" > /dev/null
            adb shell "cd ${root_dir} && \
                        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
                        ./runTrainDemo.out MemTimeProfile ${model} ../dataset/ ../dataset/train.txt ${batch} ${batch} ${target}" \
                         > "${output_dir}/${model}.${batch}.${target}.out"

            echo "finish ${model} ${batch} ${target}"
        done
    done
done