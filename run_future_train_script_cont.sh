python main_futures.py --env-name FuturesEnvCont-v0 \
  --algo a2c --num-processes 4 --recurrent-policy \
  --hidden-size 64 --activation-type relu \
  --log-interval 10 --num-steps 100  --num-updates 400
