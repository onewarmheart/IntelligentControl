from __future__ import division

import numpy as np
import os
import gym

from train import training, her_training
from evaluate import evaluating
import trainer
import buffer
from arguments import get_args
import torch
import time
import random
from datetime import datetime

import logging



def run(args, env, agent, ram, env_params):
    if args.mode == 'train' and (not args.HER):
        training(args, env, agent, ram, env_params)
    elif args.mode == 'train' and args.HER:
        her_training(args, env, agent, ram, env_params)
    else:
        evaluating(args, env, agent, args.max_episodes - 1)



if __name__ == "__main__":
    
    # get params
    args = get_args()
    
    # create save directory
    dt = datetime.fromtimestamp(int(time.time()))
    args.dir_name = dt.strftime('%Y%m%d%H%M') + '_' + args.dir_name
    os.mkdir(args.dir_name)
    
    # create environment
    env = gym.make(args.env_name)
    if args.unwrapped: env = env.unwrapped
    env_params = {
        'state_dim' : env.observation_space.shape[0],
        'action_dim' : env.action_space.shape[0],
        'action_max' : float(env.action_space.high[0])
    }
    print(' State Dimensions :- ', env_params['state_dim'])
    print(' Action Dimensions :- ', env_params['action_dim'])
    print(' Action Max :- ', env_params['action_max'])
    
    # fix seed
    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        
    # initialize memory buffer
    ram = buffer.MemoryBuffer(args.buffer_size)
    
    # initialize agent
    agent = trainer.Trainer(args,
                            env_params['state_dim'] if not args.HER else env_params['state_dim']*2,
                            env_params['action_dim'],
                            env_params['action_max'], ram)
    
    # config logging
    logging.basicConfig(filename=args.dir_name + '/log.log',level=logging.DEBUG,filemode='w')
    
    try:
        run(args, env, agent, ram, env_params)
    except:
        # end logging
        logging.shutdown()
    else:
        print("Completed!")

