# How to record an application's FPS (Frames per Second) data on Android devices without root access

When conducting experiment on our training script's effect on mobile application's performance, we hit a barrier because conventional method to dump frame rate requires root access to the phone; however, it is very hard if not impossible to get root on US versions of Samsung S25 Ultra because Samsung locked OEM settings. As a result, we need to acquire information needed using dumpsys. Let's start with an example using Genshin Impact.

## Grepping SurfaceFlinger data
Android's built-in tool dumpsys SurfaceFlinger provides detailed info about graphical surfaces and rendering stats. We need to locate the exact surface needed to fetch frame rendering data. To do that, run:

`dumpsys SurfaceFlinger | grep -i Genshin`

This command returns the layers associated with Genshin Impact (by matching name). What we need is the one that contains (BLAST). For example, the output might look like this:
```
$ dumpsys SurfaceFlinger | grep -i Genshin
  Layer [4012] Background for SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0#4012
  Layer [4011] SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0(BLAST)#4011
  Layer [4006] com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity$_30153#4006
  Layer [4006] com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity$_30153#4006
  Layer [4011] SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0(BLAST)#4011
  Layer [4012] Background for SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0#4012
  Layer [3885] 5a0597c ActivityRecordInputSink com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity#3885
 │  │  │     │  │  │     ├─ 8a8c8ff com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity#3887
 │  │  │     │  │  │     │  ├─ com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity$_30153#4006 requestedFrameRate: {0.00 Hz FrameRateCompatibility::Default FrameRateCategory::NoPreference}
 │  │  │     │  │  │     │  │  ├─ (Relative) SurfaceView[com.miHoYo.GenshinImpact[...]o.GetMobileInfo.MainActivity]@0#4010 parent=4009
 │  │  │     │  │  │     │  │  │  └─ SurfaceView[com.miHoYo.GenshinImpact[...]bileInfo.MainActivity]@0(BLAST)#4011
 │  │  │     │  │  │     │  │  └─ Bounds for - com.miHoYo.GenshinImpac[...]Yo.GetMobileInfo.MainActivity@1#4009
 SurfaceView[com.miHoYo.GenshinImpact[...]bileInfo.MainActivity]@0(BLAST)#4011
 0000000000000000 | 00 | -           |    0.0     0.0  2340.0  1080.0 |    0     0  2340  1080 |    0 |   0 | Background for SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0#4012
 000075c90000008f | 00 | RGBA_8888   |    0.0     0.0  1920.0   886.0 |    0     0  2340  1080 |    0 |   0 | SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0(BLAST)#4011
 000075c900000082 | 00 | RGBA_8888   |    0.0     0.0  2340.0  1080.0 |    0     0     0     0 |    0 |   0 | com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity$_30153#4006
+ 8a8c8ff com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity#3887 uid=1000 id=3887
+ 5a0597c ActivityRecordInputSink com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity#3885 uid=1000 id=3885
+ ActivityRecord{ed625a u0 com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity t140}#3876 uid=1000 id=3876
+ Background for SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0#4012 uid=10363 id=4012
+ SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0(BLAST)#4011 uid=10363 id=4011
+ SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0#4010 uid=10363 id=4010
+ Bounds for - com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity@1#4009 uid=10363 id=4009
+ com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity$_30153#4006 uid=10363 id=4006
  - Output Layer 0xb400006e406e48e0(SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0(BLAST)#4011)
layer: 1107 name:     SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0#7(BLAST Consumer)7 z: 0 composition: 2/2 alpha: 65535 format:         RGBA_8888_UBWC dataspace:0x08810000 transform: 90/0/0 buffer_id: 0xb4000074a8a19fb0 secure: 0

...
```
What we want is `SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0(BLAST)#4011`

We will be using this surface to probe the actual frame rate data. Use this command to dump frame timestamp data 
```
adb shell dumpsys SurfaceFlinger --latency "SurfaceView[com.miHoYo.GenshinImpact/com.miHoYo.GetMobileInfo.MainActivity]@0(BLAST)#4011"
```

The output typically looks like this:
```
16666666
11704201475481  11704226283137  11704220980012
11704233819856  11704259566262  11704253653189
11704266841627  11704292850950  11704286300221
...
```

Each line (after the first) contains three timestamps (in nanoseconds):
1. Frame start time
2. VSYNC (vertical sync) time
3. Frame ready (or presented) time

Using these timestamps, you can compute frame durations and infer FPS over time.

## Python code to convert data to frame rate
```
import numpy as np

data = """
11704201475481  11704226283137  11704220980012
11704233819856  11704259566262  11704253653189
11704266841627  11704292850950  11704286300221
...

"""  # Paste your own data

frame_times = [int(line.split()[0]) for line in data.strip().splitlines()]
intervals = np.diff(frame_times)
avg_interval = np.mean(intervals)

fps = 1_000_000_000 / avg_interval
print(f"Average FPS: {fps:.2f}")
```

Basically you just need the first column to do the calculations
## Shell script to periodically log frame rate data of a device