# Melon

## Setup

### Hardware

We used 4 devices to conduct our evaluation, and here we use **Samsung Note10** and MacBook to build and evaluate our system.

- MacBook Pro: OS X 10.15 (x86_64-apple-darwin19.6.0)
- Samsung Note10:
  - Snapdragon 855 SoC
  - 8GB main memory (**min request**)
  - 64-bit Android OS

Note that to evaluate the energy consumption of `Melon`, you **MUST** have root access to the phone.

### Software

We list the software we used:

- NDK version: r21b (Please make sure the environment variable `$ANDROID_NDK` is set properly, i.e., `ANDROID_NDK = ~/Library/Android/android-ndk-r21b`)
- Cmake version: 3.18.4
- GCC/G++ version: Apple clang version 12.0.0 (clang-1200.0.32.29)
- ADB: integrated with Android SDK, and its path is `<SDK_HOME>/platform-tools/adb`
- Python3: to plot figures, please make sure that the following packages are installed via `conda` or `pip`
  - matplotlib
  - pandas
  - numpy
  - more-itertools





## Build

Please follow the building instructions of  [MNN](https://www.yuque.com/mnn/en/build_android) to build the system.

```shell
git clone https://github.com/qipengwang/Melon.git
cd </path/to>/Melon
./schema/generate.sh
# the experiment dir contains the build scripts and other data needed
cp -r experiment/ build_64/
cd build_64/
unzip dataset.zip
./build_64.sh # build system 
```



## Run

**IMPORTANT: Please make sure that the phone does not reduce its SoC's frequency due to DVFS during running, otherwise the result may be inaccurate. You may do this by keep the  phone homothermal !**

Please connect phone via USB and enable the `Developer Option`. Please make sure that `adb shell` works:

```shell
 ~/ adb shell
d1q:/ $ mkdir -p /data/local/tmp/build_64/
d1q:/ $ ls
acct               init                  oem              
bugreports         init.environ.rc       product          
                   ...
efs                mnt                   vendor           
etc                odm                   
d1q:/ $
```



### Test run

```shell
# current dir is <Melon>/build_64/
adb shell "mkdir -p /data/local/tmp/build_64/"

# push files to phone
adb push ./*.* /data/local/tmp/build_64/ > /dev/null
adb push tools/ /data/local/tmp/build_64/ > /dev/null
adb push ./dataset/ /data/local/tmp/ > /dev/null

adb shell

# the following commands after $ are executed in adb shell
d1q:/ $ cd /data/local/tmp/build_64
d1q:/data/local/tmp/build_64 $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/                              
d1q:/data/local/tmp/build_64 $ ./runTrainDemo.out                                                                   
Usage: ./runTrainDemo.out CASENAME [ARGS]
Valid Case: 
DataLoaderDemo
DataLoaderTest
...
TestMSE
d1q:/data/local/tmp/build_64 $ 
```



### Decision Stage

The decision stage profiles the training process and generates execution plan. 

#### Profile Execution

`Melon` runs a profiling iteration to get the whole information of model, including operators' information, tensors' information, etc. Please run the following command to run the profile execution

```shell
## current dir is <Melon>/build_64/

#  By default, this command profiles the 4 models evaluated in the paper, you may need to modify the $models variable in the script if you wand to profile less models.
./profile_info.sh
```

 

#### Generate Execution Plan

After finishing the profile execution, `Melon` generates the execution plan based on the profiled information. Please run the following command to generate the execution plan.

```shell
## current dir is <Melon>/build_64/
#  By default, this command generate plans for the 4 models evaluated in the paper, you may need to modify the $models variable if you wand to generate less plans.
./generate_plan.sh
```



### Paper Experiments

The execution stage performs training guided by execution plans, which are generated through previous steps.

The running steps to reproduce the evaluation results are listed in the following part of this section.

All of the output are save to `<Melon>/build_64/data/evaluation/<EXPERIMENT>`

#### Max batch size experiment

For the end to end maximal batch size experiment, please run:

```shell
## current dir is <Melon>/build_64/
./experiment_maxbs.sh mnn
./experiment_maxbs.sh ours
python plot/plot_maxbs.py
```

Please run this experiment first because the results are reused by the following experiment.

#### Throughput experiment

For the end to end throughput experiment

- For models with BN layers, please run:

```shell
## current dir is <Melon>/build_64/
./experiment_throughput.sh mnn
./experiment_throughput.sh ours
python plot/plot_throughput.py
```

- For models without BN layers, please run:

```shell
## current dir is <Melon>/build_64/
./experiment_throughput_nobn.sh mnn
./experiment_throughput_nobn.sh ours
python plot/plot_throughput_nobn.py
```

#### Ablation

For the ablation experiment, please run:

```shell
## current dir is <Melon>/build_64/
./experiment_ablation.sh mnn
./experiment_ablation.sh ours recompute
./experiment_ablation.sh ours pool
python plot/plot_ablation.py
```

#### Adaptiveness

Note that the `Melon`'s adaptiveness is implemented based on the `realloc` function. However, the behavior of `realloc` may be different between devices, according to the [cppreference](https://en.cppreference.com/w/c/memory/realloc). 

In our experiment, `realloc` may lead to `Page Fault` when the allocated and reallocated memory size are too large, because the `pointer` returned by `realloc` is not same as the allocated one. In such case, the technique doesl not work properly.

We simulate the `Adaptiveness` process here to make sure that the experiment runs properly. Please run:

```shell
 #current dir is <Melon>/build_64/
 python plot/plot_adaptive.py
```

#### Energy consumption

**IMPORTANT**: please make sure that you have ROOT access to the phone. 

There are several steps to run this experiment, the step-by-step instructions are listed as following:

1. please check the shell `energy.sh` to make it compatible with your phone. 

   Because the vFS of difference devices may be different, the files to read may be different. For instance, the input current of usb is recorded in `/sys/class/power_supply/usb/current_now` for XiaoMi 11 Pro, while it is recorded in `/sys/class/power_supply/usb/input_current_now` for Meizu 16t. 

   So please check the vFS of your phone first by running the following steps:

    `mars:/ #` indicates that you have the ROOT access.

      ```shell
      # start adb shell
      adb shell

      mars:/ $ su
      mars:/ # ls /sys/class/power_supply/usb
      current_max  input_current_limit  subsystem  usb_type              wakeup86
      current_now  online               temp       voltage_max
      device       power                type       voltage_now
      hwmon3       quick_charge_type    uevent     waiting_for_supplier
      mars:/ # ls /sys/class/power_supply/battery
      capacity                  health      time_to_empty_avg
      charge_control_limit      hwmon0      time_to_full_avg
      charge_control_limit_max  model_name  type
      charge_counter            power       uevent
      charge_full               power_avg   voltage_max
      charge_full_design        power_now   voltage_now
      charge_type               present     voltage_ocv
      constant_charge_current   status      waiting_for_supplier
      current_now               subsystem   wakeup85
      cycle_count               technology
      device                    temp
      mars:/ # 
      ```
   
    In this example, the file we used are as following. Please modify Line10-13 of `<Melon>/build_64/energy.sh` according to your output:

      - `/sys/class/power_supply/usb/current_now`
      - `/sys/class/power_supply/usb/voltage_now`
      - `/sys/class/power_supply/battery/current_now`
      - `/sys/class/power_supply/battery/voltage_now`

2. After modifying the script, push it to the phone:

      ```shell
      ## current dir is <Melon>/build_64/
      adb push energy.sh /data/local/tmp/build_64/ > /dev/null
      ```

3. Please run the following command in an adb shell, to continuously cat the vFS file content.

      ```shell
      ## current dir is <Melon>/build_64/
      # start an adb shell, this command runs on PC
      adb shell                                                          ~/Desktop
      # adb shell starts, and the following commands runs in the adb shell to cat the vFS state file to log
      d1q:/ $ cd /data/local/tmp/build_64
      d1q:/data/local/tmp/build_64 $ su
      d1q:/data/local/tmp/build_64 # ./energy.sh
      ```
      
4. Please run the following command in **another** shell, to train a model.

    ```shell
    ## current dir is <Melon>/build_64/
    ./experiment_energy.sh mnn
    ./experiment_energy.sh ours
    ```

    During the training process, there will be multiple lines of output in the adb shell terminal of Step3.  

5. Please kill the process in the first step by `CTRL + c` after finishing the training process, and pull the log file to PC.

    ```shell
    ## current dir is <Melon>/build_64/
    adb pull /data/local/tmp/build_64/energy.out ./
    python plot/plot_energy.py
    ```



#### FL experiment

Please check the [FL repo](https://github.com/chaojin0310/Federated-learning-pytorch-cifar) for details of our FL experiment.



## Connect

If you have any questions about this repository, please email [Qipeng Wang](https://qipengwang.github.io/) via `wangqipeng AT stu DOT pku DOT edu DOT cn`.



## Citation

```
@inproceedings{wang2022melon,
  title={Melon: Breaking the memory wall for resource-efficient on-device machine learning},
  author={Wang, Qipeng and Xu, Mengwei and Jin, Chao and Dong, Xinran and Yuan, Jinliang and Jin, Xin and Huang, Gang and Liu, Yunxin and Liu, Xuanzhe},
  booktitle={Proceedings of the 20th Annual International Conference on Mobile Systems, Applications and Services},
  pages={450--463},
  year={2022}
}
```



