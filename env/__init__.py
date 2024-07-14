from gym.envs.registration import register
from env.cloth_env import ClothEnv

register(
    id='DualRLenv-v0',
    entry_point='env:ClothEnv',)

