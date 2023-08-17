#!/bin/bash

#models=("Squeezenet" "Googlenet" "MobilenetV1" "MobilenetV2")
models=("MobilenetV2" "Squeezenet")
root_dir="/data/local/tmp/build_64"

if [ $# -ne 1 ]; then
    echo "illdgal parameter, usage: ./experiment_maxbs.sh METHOD, where METHOD in {mnn, ours}"
    exit
fi
method=$1

for model in ${models[@]}; do
    mkdir -p "output/${method}/${model}/"
    if [[ $method == "mnn" ]]; then
        batch=32
        echo $model $batch $method
        adb shell "rm ${root_dir}/memory_profile.out" > /dev/null
        adb shell "cd ${root_dir} && \
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
                    ./runTrainDemo.out MemTimeProfile ${model} ../dataset/ ../dataset/train.txt ${batch} ${batch} ${method}" \
                    >  "output/${method}/${model}/${model}.${batch}.${method}.out" 2>&1
    elif [[ $method == "ours" ]]; then
        mem_bgt=5500

        if [[ $model == "MobilenetV2" ]]; then
            batchs=(64 80 96 112 128 144 160 176 192 208)
        elif [[ $model == "Squeezenet" ]]; then
            batchs=(64 80 96 112 128 144 160 176 192)
        fi

        for batch in ${batchs[@]}; do
            echo $model $batch $method $mem_bgt
            adb shell "rm ${root_dir}/memory_profile.out" > /dev/null
            adb shell "cd ${root_dir} && \
                        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
                        ./runTrainDemo.out MemTimeProfile ${model} ../dataset/ ../dataset/train.txt ${batch} ${batch} ${method} ${mem_bgt}" \
                        > "output/${method}/${model}/${model}.${batch}.${mem_bgt}.${method}.out" 2>&1
        done
    fi
done