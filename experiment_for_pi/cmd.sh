cd /data/local/tmp/build_64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/
#./runTrainDemo.out MobilenetV2Train ../dataset/ ../dataset/train.txt ../dataset/ ../dataset/train.txt
#time ./runTrainDemo.out GeneratePlan MobilenetV2 96 2000
./runTrainDemo.out MemTimeProfile  MobilenetV2 ../dataset/ ../dataset/train.txt 16 16 memcpy

#modelname == "MobilenetV2" || modelname == "MobilenetV1" || modelname == "Alexnet"
# || modelname == "Squeezenet" || modelname == "Googlenet" || modelname == "Xception" || modelname == "Resnet50"
# modelname == "MobilenetV2NoBN" || modelname == "MobilenetV1NoBN" || modelname == "SqueezenetNoBN"
#./runTrainDemo.out MnistTrain ../dataset/
#./build.sh && \
#adb push ../build/ /data/local/tmp/ && \
#adb shell 'cd /data/local/tmp/build && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:./tools/train/ && ./runTrainDemo.out MemTimeProfile  Googlenet ../dataset/ ../dataset/train.txt 16 4' > test.txt
