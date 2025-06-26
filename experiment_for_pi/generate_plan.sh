#!/bin/bash

models=("MobilenetV2" "Squeezenet" "MobilenetV1" "Resnet50")
root_dir="."

# Run cmake
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DMNN_USE_LOGCAT=OFF \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=OFF \
    -DMNN_BUILD_TRAIN=ON \
    -DMNN_BUILD_BENCHMARK=ON \
    -DMNN_OPENCL=OFF \
    -DPROFILE_EXECUTION_IN_LOG=OFF \
    -DDEBUG_EXECUTION_DETAIL=OFF \
    -DPROFILE_COST_IN_LOG=OFF \
    ../

make -j8

# Copy compiled files and tools to build directory
# cp ./*.* "${root_dir}/"
cp -r tools/ "${root_dir}/"

# Prepare directories for each model
for model in "${models[@]}"; do
    mkdir -p "${root_dir}/data/profiler/${model}"
    mkdir -p "${root_dir}/data/heu_info/${model}"
    mkdir -p "${root_dir}/heuristic/execution/${model}"
    mkdir -p "${root_dir}/heuristic/allocation/${model}"
done

# Execute GeneratePlan for each configuration
for budget in 8 6; do
    for model in "${models[@]}"; do
        if [[ $model == "MobilenetV1" && $budget -eq 8 ]]; then
            batchs=(32 64 96 128)
        elif [[ $model == "Resnet50" && $budget -eq 8 ]]; then
            batchs=(32 64 96)
        elif [[ $model == "MobilenetV2" && $budget -eq 8 ]]; then
            batchs=(40 64 80 96 112 128 144 160 176 192 208)
        elif [[ $model == "Squeezenet" && $budget -eq 8 ]]; then
            batchs=(48 64 80 96 112 128 144 160 176 192)
        elif [[ $model == "MobilenetV2" && $budget -eq 6 ]]; then
            batchs=(40 48 64 80 96 112 128)
        elif [[ $model == "Squeezenet" && $budget -eq 6 ]]; then
            batchs=(48 64 80 96 112 128)
        else
            continue
        fi

        for batch in "${batchs[@]}"; do
            echo "${model} ${batch} ${budget}"
            # Run GeneratePlan with and without recompute flag
            ./runTrainDemo.out GeneratePlan "${model}" "${batch}" "${budget}" true
            ./runTrainDemo.out GeneratePlan "${model}" "${batch}" "${budget}"
        done
    done
done

# Output is already in ./build_local/heuristic/
echo "Done. Heuristic data saved to ${root_dir}/heuristic/"
