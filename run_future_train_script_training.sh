python main_futures.py --env-name FuturesEnvTraining-v0 \
  --algo a2c --num-processes 4 --recurrent-policy \
  --hidden-size 64 --activation-type relu \
  --log-interval 100 --num-steps 10  --num-updates 100000
