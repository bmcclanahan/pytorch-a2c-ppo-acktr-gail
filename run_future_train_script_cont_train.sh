python main_futures.py --env-name FuturesEnvContTraining-v0 \
  --algo a2c --num-processes 8 --recurrent-policy \
  --hidden-size 128 --activation-type relu \
  --log-interval 2000 --num-steps 10  --num-updates 1000000 \
  --validation-interval 10000 --save-interval 10000 \
  --validation-dataset datasets/S_and_P_val.parquet \
  --training-dataset datasets/S_and_P_train.parquet
