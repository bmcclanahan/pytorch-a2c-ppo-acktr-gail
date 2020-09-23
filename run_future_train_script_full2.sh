python main_futures.py --env-name FuturesEnvFull-v1 \
  --algo a2c --num-processes 4 --recurrent-policy \
  --hidden-size 64 --activation-type relu \
  --log-interval 10 --num-steps 400  --num-updates 1000 \
  --save-dir /Users/brianmcclanahan/git_repos/AllAboutFuturesRL/historical_index_data/models2
