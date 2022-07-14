CUDA_VISIBLE_DEVICES=0 python3 train.py --dir /dataset/48g_dataset/ --data ssd_data.pt --model ssd_model_135.pt --epoch $1 --batch $2
