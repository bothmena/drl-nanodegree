from collections import deque
from typing import List
import os

import numpy as np
import torch
from unityagents import UnityEnvironment

from dqn_agent import Agent
from utils import Experience, device
from model import DQN


class DQNTrainer:
    def __init__(self,
                 env_filename: str,
                 n_episodes: int = 2000,
                 max_t: int = 1000,
                 eps_start: int = 1.0,
                 eps_end: int = 0.01,
                 eps_decay: int = 0.995,
                 save_every: int = 100,
                 target_score: float = 13.0,
                 ):
        """Deep Q-Learning.

        Params
        ======
            env_filename: path to the unity env file.
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of time-steps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            save_every: save the model every {}
            target_score: score that needs to be reached to consider the problem as solved.
        """
        self.env_filename = env_filename
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.save_every = save_every
        self.target_score = target_score

        self.agent = None
        self.scores = None
        self.scores_window = None
        self.eps = None
        self.env = None
        self.brain_name = None
        self.action_size = None
        self.state_size = None
        self.solved_weights = None

    def init_env(self):
        self.env = UnityEnvironment(file_name=self.env_filename)
        self.brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        self.state_size = len(env_info.vector_observations[0])

    def instantiate_agent(self, hidden_layers: List[int], **kwargs):
        self.agent = Agent(state_size=self.state_size, action_size=self.action_size, hidden_layers=hidden_layers, **kwargs)
        self.agent.init_dqn_networks()

    def train_dqn(self):
        self.init_training()
        for i_episode in range(1, self.n_episodes + 1):
            self.play_episode()
            self.update_eps()
            self.log(i_episode)

            if i_episode % self.save_every == 0:
                self.save_model('checkpoint_{}.pth'.format(i_episode))
            if np.mean(self.scores_window) >= self.target_score:
                avg_score = np.mean(self.scores_window)
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, avg_score))
                self.solved_weights = 'solved_{}_score_{:.2f}.pth'.format(self.agent.model_id, avg_score)
                self.save_model(self.solved_weights)
                break

    def init_training(self):
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores
        self.update_eps()

    def play_episode(self):
        state = self.env.reset(train_mode=True)[self.brain_name].vector_observations[0]
        score = 0
        for _ in range(self.max_t):
            experience = self.play_step(state)
            self.agent.step(experience)
            state = experience.next_state
            score += experience.reward
            if experience.done:
                break

        self.save_score(score)

    def play_step(self, state, inference: bool = False) -> Experience:
        if inference:
            action = self.act(state)
        else:
            action = self.agent.act(state, self.eps)
        brain_info = self.env.step(action)[self.brain_name]
        next_state = brain_info.vector_observations[0]
        reward = brain_info.rewards[0]
        done = brain_info.local_done[0]

        return Experience(state, action, reward, next_state, done)

    def save_score(self, score):
        self.scores_window.append(score)
        self.scores.append(score)

    def update_eps(self):
        if self.eps is None:
            self.eps = self.eps_start  # init epsilon
        else:
            self.eps = max(self.eps_end, self.eps_decay * self.eps)  # decrease epsilon

    def log(self, episodes: int):
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episodes, np.mean(self.scores_window)), end="")
        if episodes % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episodes, np.mean(self.scores_window)))

    def save_model(self, filename: str, directory: str = 'saved_weights'):
        torch.save(self.agent.dqn_local.state_dict(), os.path.join(directory, 'local_{}'.format(filename)))
        torch.save(self.agent.dqn_target.state_dict(), os.path.join(directory, 'target_{}'.format(filename)))

    def set_trained_dqn(self, weights_path: str = None):
        weights_path = self.solved_weights if weights_path is None else weights_path
        if weights_path is None:
            raise ValueError('please provide the path to the trained model weights.')
        if self.agent.dqn_local is None:
            self.agent.dqn_local = DQN(self.state_size, self.action_size, self.agent.hidden_layers)
            self.agent.init_dqn_networks(inference=True)
        self.agent.dqn_local.load_state_dict(torch.load(weights_path))

    def play(self):
        self.eps = 0.
        env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        timestep = 0
        while True:
            timestep += 1
            experience = self.play_step(state, inference=True)
            state = experience.state
            score += experience.reward
            print('\rTime step {}\tScore: {:.2f}'.format(timestep, score), end="")
            if experience.done:
                break

        print("\rFinal Score: {:.2f}".format(score))

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.agent.dqn_local.eval()
        with torch.no_grad():
            action_values = self.agent.dqn_local(state)

        return np.argmax(action_values.cpu().data.numpy())
