# Useful Commands

## Monitor GPU Usage
This command monitors GPU usage and updates every second (only the bottom processes part).
```sh
watch -n 1 "nvidia-smi | awk '/Processes:/ {flag=1; next} flag'"
```

### Monitor GPU Usage with nvitop
```
pip install nvitop
nvitop
```


## List Open Files on NVIDIA Devices
This command lists all open files on NVIDIA devices.
```sh
lsof /dev/nvidia*
```


## To Check Running Processes
You can monitor the running process using:
```sh
ps aux | grep dist_train.sh
ps aux | grep train.py
pgrep -laf train.py
```

## To Kill a Process
You can kill a process using:
```sh
kill -9 <PID>
```
Where `<PID>` is the process ID you want to kill.


## Plot Loss Curves
This command plots loss curves from the log file.
```sh
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_combined/20250219_183536/vis_data/20250219_183536.json  --keys loss_cls loss_bbox loss_iou --legend Classification_Loss Bounding_Box_Loss IOU_Loss --title Loss_Curves --out combined_loss_plot.png
```