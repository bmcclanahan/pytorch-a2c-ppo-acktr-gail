python main_futures.py --env-name FuturesEnvContTraining-v0 \
  --algo acktr --num-processes 4  \
  --hidden-size 64 --activation-type relu \
  --log-interval 10 --num-steps 100  --num-updates 8000
