python main_futures.py --env-name FuturesEnv-v0 \
  --algo a2c --num-processes 4 --recurrent-policy \
  --hidden-size 64 --activation-type relu \
  --log-interval 10 --num-steps 10  --num-updates 300 --save-interval 1400
