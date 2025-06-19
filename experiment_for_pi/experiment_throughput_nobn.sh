#!/bin/bash

#models=("Squeezenet" "Googlenet" "MobilenetV1" "MobilenetV2")
models=("MobilenetV2NoBN" "SqueezenetNoBN" "MobilenetV1NoBN" "Resnet50NoBN")
root_dir="/data/local/tmp/build_64"

if [ $# -ne 1 ]; then
    echo "illdgal parameter, usage: ./experiment_maxbs.sh METHOD, where METHOD in {mnn, ours}"
    exit
fi
method=$1
micro=16


for model in ${models[@]}; do
    mkdir -p "output/${method}/${model}/"
    if [[ $method == "mnn" ]]; then
        echo $model $micro $method
        adb shell "rm ${root_dir}/memory_profile.out" > /dev/null
        adb shell "cd ${root_dir} && \
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
                    ./runTrainDemo.out MemTimeProfile ${model} ../dataset/ ../dataset/train.txt ${micro} ${micro} ${method}" \
                    > "output/${method}/${model}/${model}.${micro}.${method}.out" 2>&1

    elif [[ $method == "ours" ]]; then

        if [[ $model == "MobilenetV1NoBN" ]] || [[ $model == "MobilenetV2NoBN" ]] || [[ $model == "SqueezenetNoBN" ]]; then
            batchs=(64 96 128)
        else
            batchs=(32 64)
        fi

        for batch in ${batchs[@]}; do
            echo $model $batch $method $mem_bgt
            adb shell "rm ${root_dir}/memory_profile.out" > /dev/null
            adb shell "cd ${root_dir} && \
                        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
                        ./runTrainDemo.out MemTimeProfile ${model} ../dataset/ ../dataset/train.txt ${batch} ${micro} mnn" \
                        > "output/${method}/${model}/${model}.${batch}.${micro}.${method}.out" 2>&1
        done
    fi
done