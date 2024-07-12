from gym.envs.registration import register
from env.cloth_env import ClothEnv

register(id='DualRLenv',
         entry_point='env:ClothEnv',)

