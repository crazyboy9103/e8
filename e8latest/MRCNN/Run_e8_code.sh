#rm mrcnn_data.pt
bn=12
ep=10
CMD="python3 train.py --dir /dataset/48g_dataset/ --data mrcnn_data.pt --model mrcnn_model_75.pt --epoch $ep --batch $bn"
echo `date` >> history_run_cmd_ssd_A6000.txt
echo $CMD
echo $CMD >> history_run_cmd_ssd_A6000.txt
time $CMD
