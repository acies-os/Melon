#!/bin/bash

#models=("Squeezenet" "Googlenet" "MobilenetV1" "MobilenetV2")
models=("MobilenetV2" "Squeezenet")
root_dir="/data/local/tmp/build_64"

adb shell "cd ${root_dir}/heuristic && rm -r *" > /dev/null
method=$1
ablation=$2

if [[ $method == "ours" && $# -ne 2 ]]; then
    echo "usage: ./experiment_ablation.sh {mnn | ours (recompute | pool)}"
fi


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
        if [[ $model == "MobilenetV2" ]] && [[ $ablation == "pool" ]]; then
            batches=(40)
        elif [[ $model == "MobilenetV2" ]] && [[ $ablation == "recompute" ]]; then
            batches=(80)
        elif [[ $model == "Squeezenet" ]] && [[ $ablation == "recompute" ]]; then
            batches=(64 80)
        else
            batches=(48)
        fi
        mem_bgt=5500
        smaller_budget=0
        for batch in ${batches[@]}; do
            if [[ $ablation == "pool" ]]; then
                adb push heuristic/allocation /data/local/tmp/build_64/heuristic > /dev/null
                adb shell "cp ${root_dir}/heuristic/allocation/${model}/${model}.${batch}.address.txt \
                              ${root_dir}/heuristic/allocation/${model}/${model}.${batch}.${mem_bgt}.address.txt" > /dev/null
            else
                adb push heuristic/execution /data/local/tmp/build_64/heuristic > /dev/null
            fi
            echo $model $batch $method $mem_bgt $ablation
            adb shell "rm ${root_dir}/memory_profile.out" > /dev/null

            adb shell "cd ${root_dir} && \
                        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && \
                        ./runTrainDemo.out MemTimeProfile ${model} ../dataset/ ../dataset/train.txt ${batch} ${batch} ${method} ${mem_bgt}" \
                        > "output/${method}/${model}/${model}.${batch}.${mem_bgt}.${method}.${ablation}.out" 2>&1
        done
    fi
done
