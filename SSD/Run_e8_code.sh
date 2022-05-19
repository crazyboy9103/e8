#rm ssd_data.pt
bn=32
ep=5
CMD="python3 train.py --dir /dataset/48g_dataset/ --data ssd_data.pt --model ssd_model_135.pt --epoch $ep --batch $bn"
echo `date` >> history_run_cmd_ssd_A6000.txt
echo $CMD
echo $CMD >> history_run_cmd_ssd_A6000.txt
time $CMD
