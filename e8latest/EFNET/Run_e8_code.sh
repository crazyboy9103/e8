CMD="python3 process_data.py"

echo $CMD
echo $CMD >> history_run_cmd_ssd_A6000.txt
time $CMD

bash train_crack.sh 
bash train_finish.sh
bash train_ground.sh
bash train_living.sh
bash train_peel.sh 
bash train_rebar.sh 
bash train_window.sh

echo `date` >> history_run_cmd_ssd_A6000.txt
