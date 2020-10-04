python main_futures.py --env-name FuturesEnvTraining-v0 \
  --algo a2c --num-processes 8 --recurrent-policy \
  --hidden-size 64 --activation-type relu \
  --log-interval 2000 --num-steps 10  --num-updates 1000000 \
  --validation-interval 10000 --save-interval 10000\
  --validation-dataset datasets/S_and_P_val.parquet \
  --training-dataset datasets/S_and_P_train.parquet #\
  --load-saved-model trained_models/a2c/FuturesEnvTraining-v0-2020-09-29-19-39-58.pt
