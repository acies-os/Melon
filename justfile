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

flags := '\
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
-DDEBUG_EXECUTION_DETAIL=OFF \
-DMNN_BUILD_BENCHMARK=ON \
-DMNN_BUILD_DEMO=ON \
-DMNN_BUILD_FOR_ANDROID_COMMAND=OFF \
-DMNN_BUILD_TRAIN=ON \
-DMNN_OPENCL=OFF \
-DMNN_USE_LOGCAT=OFF \
-DNATIVE_INCLUDE_OUTPUT=. \
-DNATIVE_LIBRARY_OUTPUT=. \
-DPROFILE_COST_IN_LOG=OFF \
-DPROFILE_EXECUTION_IN_LOG=OFF \
'

# ------------------------------ build commands ------------------------------

# flatbuffer code generation
schema-gen:
    ./schema/generate.sh

build j=default-j:
    mkdir -p build && \
    cd build && \
    cmake .. {{ mac-m1-flag }} {{ flags }} && \
    make -j{{ j }}

clean:
    rm -rf build/

clean-all: clean
    rm -rf ./3rd_party/flatbuffers/tmp
