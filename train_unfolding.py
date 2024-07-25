import mujoco_py
import torch
import gym
from utils import general_utils
import copy
import numpy as np
from env import cloth_env
import logging
import time

from rlkit.torch import pytorch_util, networks, torch_rl_algorithm
from rlkit.torch.sac import policies as sac_policies, sac
from rlkit.torch.her.cloth import her
from rlkit.launchers import launcher_util
from rlkit.envs import wrappers

from rlkit.samplers.eval_suite import success_rate_test, eval_suite, real_corner_prediction_test
#************************************#
from rlkit.samplers.eval_suite import folding_test
from rlkit.samplers import data_collector
from rlkit.data_management import future_obs_dict_replay_buffer

#**********바꾼부분: CUDA error*******************#
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


torch.cuda.empty_cache()
gym.logger.set_level(50)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def experiment(variant):
    #************************************#
    eval_env = cloth_env.ClothEnv(
        **variant['env_kwargs'], randomization_kwargs=variant['randomization_kwargs'])
    # eval_env = gym.make('DualRLenv-v0',**variant['env_kwargs'],randomization_kwargs=variant['randomization_kwargs'])
    

    randomized_eval_env = general_utils.get_randomized_env(
        wrappers.NormalizedBoxEnv(eval_env), randomization_kwargs=variant['randomization_kwargs'])

    env_keys, env_dims = general_utils.get_keys_and_dims(
        variant, randomized_eval_env)

    fc_width, fc_depth = variant['value_function_kwargs']['fc_layer_size'], variant['value_function_kwargs']['fc_layer_depth']

    qf1 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    qf2 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    target_qf1 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    target_qf2 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )

    policy = sac_policies.TanhScriptPolicy(
        output_size=env_dims['action_dim'],
        added_fc_input_size=env_dims['added_fc_input_size'],
        **variant['policy_kwargs'],
    )

    eval_policy = sac_policies.MakeDeterministic(policy)

    success_test = success_rate_test.SuccessRateTest(
        env=randomized_eval_env,
        policy=eval_policy,
        keys=env_keys,
        name='randomized_cloth',
        metric_keys=['success_rate', 'corner_distance', 'corner_0',
                     'corner_1', 'corner_2', 'corner_3', 'corner_sum_error'],
        **variant['eval_kwargs'],
    )

    #********바꾼부분*******************#
    evaluation_suite = eval_suite.EvalTestSuite(
        tests=[success_test])

    def make_worker_env_function():
        return general_utils.get_randomized_env(wrappers.NormalizedBoxEnv(cloth_env.ClothEnv(**variant['env_kwargs'], randomization_kwargs=variant['randomization_kwargs'])), randomization_kwargs=variant['randomization_kwargs'])

    env_functions = [make_worker_env_function for _ in range(
        variant['path_collector_kwargs']['num_processes'])]
    vec_env = wrappers.SubprocVecEnv(env_functions)

    exploration_path_collector = data_collector.VectorizedKeyPathCollector(
        vec_env,
        policy,
        observation_key=env_keys['path_collector_observation_key'],
        desired_goal_key=env_keys['desired_goal_key'],
        **variant['path_collector_kwargs'],
    )

    replay_buffer = future_obs_dict_replay_buffer.FutureObsDictRelabelingBuffer(
        ob_spaces=copy.deepcopy(eval_env.observation_space.spaces),
        action_space=copy.deepcopy(eval_env.action_space),
        task_reward_function=randomized_eval_env.task_reward_function,
        observation_key=env_keys['observation_key'],
        desired_goal_key=env_keys['desired_goal_key'],
        achieved_goal_key=env_keys['achieved_goal_key'],
        **variant['replay_buffer_kwargs']
    )

    trainer = sac.SACTrainer(
        policy_target_entropy=-np.prod(
            eval_env.action_space.shape).item(),
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    trainer = her.ClothSacHERTrainer(trainer)

    algorithm = torch_rl_algorithm.TorchBatchRLAlgorithm(
        eval_suite=evaluation_suite,
        trainer=trainer,
        exploration_data_collector=exploration_path_collector,
        replay_buffer=replay_buffer,
        env_dims=env_dims,
        policy_kwargs=variant['policy_kwargs'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(pytorch_util.device)

    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

    #************************************#
    #240712
    # for episode in range(5):
    #     obs=randomized_eval_env.reset()
    #     randomized_eval_env.setup_viewer()
    #     for t in range(20):
    #         print("tried to visualize simulator environment with render")
    #         randomized_eval_env.viewer.render(600,600)
    #         action = randomized_eval_env.action_space.sample() #여기서 action 뽑아내는 거를 policy를 이용해서 뽑아야 할거같은데?
    #         observation, reward, done, info = randomized_eval_env.step(action)
    #         time.sleep(0.1)
    #         print(observation, reward, done, info)
    #         # epoch, cycle등을 작게 했을 때는 reward가 계속 -1로, improve되지 않았음
    #         if done:
    #             break
    # randomized_eval_env.close()
    #

    vec_env.close()
    logger.debug("Closed subprocesses: make_worker_env_function()")
    return


if __name__ == "__main__":
    args = general_utils.argsparser()
    """
    Namespace(albumentations_randomization=1, 
                batch_size=256, buffer_size=100000.0, 
                camera_config='small', camera_position_randomization=1, camera_type='side', 
                cloth_size=0.2, control_frequency=10, corner_prediction_loss_coef=0.001, 
                damping_ratio=1, discount=0.99, dynamics_randomization=1, 
                extra_reward=1, fail_reward=-1, fc_layer_depth=5, fc_layer_size=512, filter=0.03, 
                fovy_perturbation_size=0.05, frame_stack_size=1, goal_noise=0.03, 
                her_percent=0.8, image_obs_noise_mean=0.5, image_obs_noise_std=0.5, 
                kp=1000.0, lights_randomization=1, lookat_position_randomization=1, lookat_position_randomization_radius=0.03, 
                materials_randomization=1, max_close_steps=10, max_path_length=50, 
                num_cycles=5, num_demoers=0, num_epochs=3, num_eval_rollouts=5, num_processes=1, 
                output_max=0.03, position_perturbation_size=0.2, 
                robot_observation='ctrl', rotation_perturbation_size=0.75, 
                run=0, save_images_every_epoch=1, save_policy_every_epoch=1, sparse_dense=1, 
                success_distance=0.05, success_reward=0, timestep=0.01, 
                title='default', train_steps=500)
    """
    
    #************************************#
    variant = general_utils.get_variant(args, algorithm="SAC")

    general_utils.setup_training_device()
    general_utils.setup_save_folder(variant)
    launcher_util.setup_logger(
        variant["title"], variant=variant, base_log_dir=variant["save_folder"])

    logger.debug('Training started')
    experiment(variant)