python main_futures.py --env-name FuturesEnvCNNPosSpace-v0 \
  --algo a2c --num-processes 8 \
  --hidden-size 128 \
  --log-interval 100 --num-steps 10  --num-updates 4000\
  --validation-interval 1000 --save-interval 10000
