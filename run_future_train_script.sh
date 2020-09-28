python main_futures.py --env-name FuturesEnv-v0 \
  --algo a2c --num-processes 4 --recurrent-policy \
  --hidden-size 64 --activation-type relu \
  --log-interval 10 --num-steps 10  --num-updates 50 --save-interval 10 \
  --load-saved-model trained_models/a2c/FuturesEnv-v0-2020-09-27-16-12-25.pt
