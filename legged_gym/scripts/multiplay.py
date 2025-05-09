import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from argparse import Namespace
from isaacgym import gymapi



def play(args):
    args1= Namespace(checkpoint=None, compute_device_id=0, 
              experiment_name=None, flex=False, 
              graphics_device_id=0, headless=False, 
              horovod=False, load_run='Apr17_18-04-53_', 
              max_iterations=None, num_envs=None, num_threads=0, 
              physics_engine=gymapi.SIM_PHYSX, physx=False, pipeline='gpu', 
              resume=False, rl_device='cuda:0', run_name=None, seed=None, 
              sim_device='cuda:0', sim_device_id=0, sim_device_type='cuda', 
              slices=0, subscenes=0, task='g1', use_gpu=True, use_gpu_pipeline=True)

    args2= Namespace(checkpoint=None, compute_device_id=0, 
              experiment_name=None, flex=False, 
              graphics_device_id=0, headless=False, 
              horovod=False, load_run='Apr22_17-17-41_', 
              max_iterations=None, num_envs=None, num_threads=0, 
              physics_engine=gymapi.SIM_PHYSX, physx=False, pipeline='gpu', 
              resume=False, rl_device='cuda:0', run_name=None, seed=None, 
              sim_device='cuda:0', sim_device_id=0, sim_device_type='cuda', 
              slices=0, subscenes=0, task='g1', use_gpu=True, use_gpu_pipeline=True)
    args3= Namespace(checkpoint=None, compute_device_id=0, 
              experiment_name=None, flex=False, 
              graphics_device_id=0, headless=False, 
              horovod=False, load_run='May03_11-17-07_', 
              max_iterations=None, num_envs=None, num_threads=0, 
              physics_engine=gymapi.SIM_PHYSX, physx=False, pipeline='gpu', 
              resume=False, rl_device='cuda:0', run_name=None, seed=None, 
              sim_device='cuda:0', sim_device_id=0, sim_device_type='cuda', 
              slices=0, subscenes=0, task='g1', use_gpu=True, use_gpu_pipeline=True)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args1.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args1.task, args=args1, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args1.task, args=args1, train_cfg=train_cfg)
    ppo_runner2, train_cfg2 = task_registry.make_alg_runner(env=env, name="g1", args=args2, train_cfg=train_cfg)
    ppo_runner3, train_cfg2 = task_registry.make_alg_runner(env=env, name="g1", args=args3, train_cfg=train_cfg)
    policy1 = ppo_runner.get_inference_policy(device=env.device)
    policy2 = ppo_runner2.get_inference_policy(device=env.device)
    policy3 = ppo_runner3.get_inference_policy(device=env.device)
    while(1):
        print("**///************************")
        # print("before1")
        for i in range(50):
            actions2 = policy2(obs.detach())
            obs, _, rews, dones, infos = env.step(actions2.detach())
        print("**///************************")
        print("end1")
        for i in range(50):
            actions = policy1(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
        print("**///************************")
        print("end2")
        for i in range(10):
            actions3 = policy3(obs.detach())
            obs, _, rews, dones, infos = env.step(actions3.detach())
        print("**///************************")
        print("end2")


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    print("//*****")
    print(args)
    print("//*****")
    play(args)
