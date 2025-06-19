#!/bin/bash

# Define models to test
models=("MobilenetV2" "Squeezenet" "MobilenetV1" "Resnet50")

# Root build/output directory
root_dir="./build_local"

# Ensure local directory structure
for model in "${models[@]}"; do
    mkdir -p "${root_dir}/heuristic/allocation/${model}"
    mkdir -p "${root_dir}/heuristic/execution/${model}"
    echo "maxsize 2500000000" > "${root_dir}/heuristic/allocation/${model}/${model}.address.txt"
    echo "0 0" >> "${root_dir}/heuristic/allocation/${model}/${model}.address.txt"
done

# Targets: profile / resize / cost
for target in "profile" "resize" "cost"; do
    if [[ $target == "profile" ]]; then
        build_opt="-DPROFILE_EXECUTION_IN_LOG=ON -DDEBUG_EXECUTION_DETAIL=OFF -DPROFILE_COST_IN_LOG=OFF"
    elif [[ $target == "resize" ]]; then
        build_opt="-DPROFILE_EXECUTION_IN_LOG=OFF -DDEBUG_EXECUTION_DETAIL=ON -DPROFILE_COST_IN_LOG=OFF"
    else
        build_opt="-DPROFILE_EXECUTION_IN_LOG=OFF -DDEBUG_EXECUTION_DETAIL=OFF -DPROFILE_COST_IN_LOG=ON"
    fi

    echo "==== Building with $build_opt ===="

    # Build with profiling flags
    cmake -B build_local -S ../ \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_USE_LOGCAT=OFF \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=OFF \
        -DMNN_BUILD_TRAIN=ON \
        -DMNN_BUILD_BENCHMARK=ON \
        -DMNN_OPENCL=OFF \
        ${build_opt}

    make -C build_local -j8

    # Profiling loop
    for model in "${models[@]}"; do
        output_dir="${target}/${model}"
        mkdir -p "$output_dir"

        # Assign batch sizes per model
        if [[ $model == "MobilenetV1" ]]; then
            batchs=(32 64 96 128)
        elif [[ $model == "Resnet50" ]]; then
            batchs=(32 64 96)
        elif [[ $model == "MobilenetV2" ]]; then
            batchs=(40 48 64 80 96 112 128 144 160 176 192 208)
        elif [[ $model == "Squeezenet" ]]; then
            batchs=(48 64 80 96 112 128 144 160 176 192)
        fi

        for batch in "${batchs[@]}"; do
            echo "Running $model with batch $batch in mode $target"
            rm -f "${root_dir}/memory_profile.out"

            # Run the profiling executable
            ./build_local/runTrainDemo.out MemTimeProfile "$model" dataset/ dataset/train.txt "$batch" "$batch" "$target" \
                > "${output_dir}/${model}.${batch}.${target}.out"

            echo "Finished $model $batch $target"
        done
    done
done
