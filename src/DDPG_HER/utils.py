import numpy as np
import torch
import shutil
import torch.autograd as Variable
import random

def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
	"""
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
	ou = OrnsteinUhlenbeckActionNoise(1)
	states = []
	for i in range(1000):
		states.append(ou.sample())
	import matplotlib.pyplot as plt

	plt.plot(states)
	plt.show()
    
# helper functions for her's training
    
def generate_goals(i, episode_cache, sample_num, sample_range = 200):
    '''
    Input: current steps, current episode transition's cache, sample number 
    Return: new goals sets
    notice here only "future" sample policy
    '''
    end = (i+sample_range) if i+sample_range < len(episode_cache) else len(episode_cache)
    epi_to_go = episode_cache[i:end]
    if len(epi_to_go) < sample_num:
        sample_trans = epi_to_go
    else:
        sample_trans = random.sample(epi_to_go, sample_num)
    return [np.array(trans[3][:3]) for trans in sample_trans]

def calcu_reward(new_goal, state, action, mode='her'):
    # direcly use observation as goal
    
    if mode == 'shaping':
        # shaping reward
        goal_cos, goal_sin, goal_thdot = new_goal[0], new_goal[1], new_goal[2]
        cos_th, sin_th, thdot = state[0], state[1], state[2]
        costs = (goal_cos - cos_th)**2 + (goal_sin - sin_th)**2 + 0.1*(goal_thdot-thdot)**2
        reward = -costs
    elif mode  == 'her':
        # binary reward, no theta now 
        tolerance = 0.5
        goal_cos, goal_sin, goal_thdot = new_goal[0], new_goal[1], new_goal[2]
        cos_th, sin_th, thdot = state[0], state[1], state[2]
        costs = (goal_cos - cos_th)**2 + (goal_sin - sin_th)**2 + 0.1*(goal_thdot-thdot)**2
        reward = 0 if costs < tolerance else -1
    return reward

def gene_new_sas(new_goals, transition):
    state, new_state = transition[0][:3], transition[3][:3]
    action = transition[1]
    state = np.concatenate((state, new_goals))
    new_state = np.concatenate((new_state, new_goals))
    return state, action, new_state

#def angle_normalize(x):
#    return (((x+np.pi) % (2*np.pi)) - np.pi)