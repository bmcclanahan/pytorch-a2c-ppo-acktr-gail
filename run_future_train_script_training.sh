python main_futures.py --env-name FuturesEnvTraining-v0 \
  --algo a2c --num-processes 4 --recurrent-policy \
  --hidden-size 64 --activation-type relu \
  --log-interval 100 --num-steps 10  --num-updates 100000 \
  --validation-interval 1000 \
  --validation-dataset datasets/S_and_P_val.parquet \
  --training-dataset datasets/S_and_P_train.parquet \
  --load-saved-model trained_models/a2c/FuturesEnv-v0-old.pt
