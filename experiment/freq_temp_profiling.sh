#!/system/bin/sh

output_file="freq_info.out"

echo "Timestamp(ms) Frequency(kHz) Temperature(C)" > "$output_file"

while true
do
    timestamp=`date +%s%3N`
    freq=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null`
    temp=`cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null`
    if [ -n "$freq" ]; then
        temp_c=$(awk "BEGIN {printf \"%.1f\", $temp/1000}")
        echo "$timestamp $freq $temp_c" >> "$output_file"
    else
        echo "$timestamp ERROR" >> "$output_file"
    fi
    sleep 0.1
done

