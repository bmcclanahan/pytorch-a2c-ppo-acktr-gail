python main_futures.py --env-name FuturesEnvFullPosSpace-v0 \
  --algo a2c --num-processes 4 --recurrent-policy \
  --hidden-size 64 --activation-type relu \
  --log-interval 2000 --num-steps 10  --num-updates 1000000 \
  --validation-interval 10000 --save-interval 10000\
  --gamma 0.9 \
  --validation-dataset datasets/S_and_P_val_2020_9.parquet \
  --training-dataset datasets/S_and_P_train_2020_9.parquet
