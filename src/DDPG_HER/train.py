# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:26:13 2019

@author: Hill
"""


import numpy as np
from plot_tools import plot_reward

import gc
import psutil
import logging 
import random

from evaluate import evaluating, her_evaluating
from utils import generate_goals, calcu_reward, gene_new_sas

def training(args, env, agent, ram, env_params):

    return_history = []
    for _ep in range(args.max_episodes):
#        if(_ep % args.evaluate_interval == 0):
#            evaluating(args, env, agent, 0)
            
        observation = env.reset()
        print('EPISODE :- ', _ep)
        ep_r = 0

        for r in range(args.max_steps):
#            env.render()
            state = np.float32(observation)
    #        state = observation
            action = agent.get_exploration_action(state)
    
            new_observation, reward, done, info = env.step(action)
            
            ep_r = ep_r + reward

    
            if done:
                new_state = None
            else:
    #            new_state = new_observation
                new_state = np.float32(new_observation)
                # push this exp in ram
                ram.add(state, action, reward, new_state)
    
            observation = new_observation
    
            # perform optimization
            agent.optimize()
            if done:
                break
        
        print("Episode: {} | Return: {}".format(_ep, ep_r))
        logging.info("Episode: {} | Return: {}".format(_ep, ep_r))
        return_history.append(ep_r)
        # check memory consumption and clear memory
        gc.collect()
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)
    
        if _ep == 0 or _ep % 99 == 0:
            agent.save_models(args.dir_name, _ep)
    plot_reward(return_history, args.dir_name)
    
    print('Completed episodes')
    
def her_training(args, env, agent, ram, env_params):
    pendulum_goal = np.array([1.0, 0.0, 0.0], dtype = np.float32)
    goals = pendulum_goal
    return_history = []
    test_return_history = []
    for _ep in range(args.max_episodes):
        if(_ep == 0 or (_ep + 1) % args.evaluate_interval == 0):
            test_return_history.append(her_evaluating(args, env, agent, 0))
        
        # PART I NORMAL INTERACTION
        observation = env.reset()
        print('EPISODE :- ', _ep)
        ep_r = 0
        episode_cache = []
        for r in range(args.max_steps):
#            env.render()
            state = np.float32(observation)
            state = np.concatenate((state, goals))
            action = agent.get_exploration_action(state)
    
            new_observation, reward, done, info = env.step(action)
            # discard original reward from gym
            reward = calcu_reward(goals, state, action)
            ep_r = ep_r + reward
    
            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)
                new_state = np.concatenate((new_state, goals))
                episode_cache.append((state, action, reward, new_state))
                # push this exp in ram
                ram.add(state, action, reward, new_state)
    
            observation = new_observation
            # not in paper's pseudo code
            agent.optimize()
            if done:
#                print("break")
                break
        
        print("Episode: {} | Return: {}".format(_ep, ep_r))
        logging.info("Episode: {} | Return: {}".format(_ep, ep_r))
        return_history.append(ep_r)
        
        # PART II hindsight replay
        for i, transition in enumerate(episode_cache):
            new_goals = generate_goals(i, episode_cache, args.HER_sample_num)
            for new_goal in new_goals:
                reward = calcu_reward(new_goal, state, action) 
                state, action, new_state = gene_new_sas(new_goal, transition)
                ram.add(state, action, reward, new_state)
        
        
        # PART III optimization
        for k in range(1):
            # perform optimization
            agent.optimize()
        
        # check memory consumption and clear memory
        gc.collect()
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)
    
        if _ep == 0 or _ep % 99 == 0:
            agent.save_models(args.dir_name, _ep)
    plot_reward(test_return_history, args.dir_name)
    
    # PART II NORMAL INTERACTION
    
    print('Completed episodes')
    
