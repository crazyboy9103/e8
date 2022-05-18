rm ssd_data.pt
bn=16
ep=8
CMD="python3 train.py --dir /dataset/48g_dataset/ --data ssd_data.pt --model ssd_model_130.pt --epoch $ep --batch $bn"
echo `date` >> history_run_cmd_ssd_A6000.txt
echo $CMD
echo $CMD >> history_run_cmd_ssd_A6000.txt
time $CMD
