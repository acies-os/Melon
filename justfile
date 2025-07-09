[private]
default:
    @just -f {{ justfile() }} --list

#
# --------------------------------- variables ---------------------------------
#
# flag to build on apple silicon

default-j := `nproc`
mac-m1-flag := if os() == 'macos' { if arch() == 'aarch64' { '-DCMAKE_OSX_ARCHITECTURES=arm64' } else { '' } } else { '' }

# other flags to build melon

build-dir := "build_64"
type := "Release"
flags := '\
-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
-DMNN_BUILD_BENCHMARK=ON \
-DMNN_BUILD_DEMO=ON \
-DMNN_BUILD_FOR_ANDROID_COMMAND=OFF \
-DMNN_BUILD_TRAIN=ON \
-DMNN_OPENCL=OFF \
-DMNN_USE_LOGCAT=OFF \
-DNATIVE_INCLUDE_OUTPUT=. \
-DNATIVE_LIBRARY_OUTPUT=. \
'
build-opt-flags := '\
-DPROFILE_COST_IN_LOG=OFF \
-DDEBUG_EXECUTION_DETAIL=OFF \
-DPROFILE_EXECUTION_IN_LOG=OFF \
'

# ------------------------------ build commands ------------------------------

# flatbuffer code generation
schema-gen:
    ./schema/generate.sh

build j=default-j:
    mkdir -p {{ build-dir }} && \
    cd {{ build-dir }} && \
    cmake .. {{ mac-m1-flag }} {{ flags }} -DCMAKE_BUILD_TYPE={{ type }} {{ build-opt-flags }} && \
    make -j{{ j }}

build-profile j=default-j:
    just build-opt-flags="-DPROFILE_EXECUTION_IN_LOG=ON -DDEBUG_EXECUTION_DETAIL=OFF -DPROFILE_COST_IN_LOG=OFF" build-dir={{ build-dir }} type={{ type }} build {{ j }}

build-resize j=default-j:
    just build-opt-flags="-DPROFILE_EXECUTION_IN_LOG=OFF -DDEBUG_EXECUTION_DETAIL=ON -DPROFILE_COST_IN_LOG=OFF" build-dir={{ build-dir }} type={{ type }} build {{ j }}

build-cost j=default-j:
    just build-opt-flags="-DPROFILE_EXECUTION_IN_LOG=OFF -DDEBUG_EXECUTION_DETAIL=OFF -DPROFILE_COST_IN_LOG=ON" build-dir={{ build-dir }} type={{ type }} build {{ j }}

clean:
    rm -rf {{ build-dir }}/

clean-all: clean
    rm -rf ./3rd_party/flatbuffers/tmp

run-exp-pi:
    cp -r experiment_for_pi/ build_64/
    cd build_64 && unzip dataset.zip
    cd build_64 && ./build_64.sh

prep-data:
    cp experiment_for_pi/dataset.zip {{ build-dir }}
    cd {{ build-dir }} && unzip dataset.zip

profile-info:
    just build-dir=build_prof_one build-profile
    just build-dir=build_prof_one prep-data
    cd ./build_prof_one && runTrainDemo.out MemTimeProfile MobilenetV2 dataset/ dataset/train.txt 64 64 profile

run-profile model batch-size target:
    cd {{ build-dir }} && \
    ./runTrainDemo.out \
    MemTimeProfile \
    {{ model }} \
    dataset/ dataset/train.txt \
    {{ batch-size }} {{ batch-size }} {{ target }} > "output.{{ model }}.{{ batch-size }}.{{ target }}.txt"

gen-clean:
    rm -rf "{{ build-dir }}/heuristic" "{{ build-dir }}/data"
    mkdir -p "{{ build-dir }}/data/profiler/MobilenetV2" \
        "{{ build-dir }}/data/heu_info/MobilenetV2" \
        "{{ build-dir }}/heuristic/execution/MobilenetV2" \
        "{{ build-dir }}/heuristic/allocation/MobilenetV2"

gen1: gen-clean
    cd {{ build-dir }} && \
    ./runTrainDemo.out GeneratePlan MobilenetV2 40 8 true

gen2:
    cd {{ build-dir }} && \
    ./runTrainDemo.out GeneratePlan MobilenetV2 40 8
