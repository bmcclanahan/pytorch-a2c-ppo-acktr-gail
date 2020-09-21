from gym.envs.registration import register

register(
    id='FuturesEnv-v0',
    entry_point='futures_env.env1:Environment',
)
