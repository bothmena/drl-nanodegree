import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from dqn_agent import Agent
from dqn_trainer import DQNTrainer
from model import DQN

env_path = "/Users/bothmena/Projects/ai/ReinforcementLearning/deep-reinforcement-learning/bin/Banana.app"
DQN_HIDDEN_LAYERS = [64, 64]  # in dqn_agent.py
# DQN_HIDDEN_LAYERS = [64, 128, 128, 128]

trainer = DQNTrainer(env_filename=env_path, target_score=13.)
trainer.init_env()
# using default values for batch size, buffer size, ...
# default values are defined in dqn_agent.py, you can override them by adding more arguments
# to the Agent class call.
trainer.instantiate_agent(DQN_HIDDEN_LAYERS)
# trainer.train_dqn()

# plt.figure(figsize=(20, 10))
# plt.plot(trainer.scores)
# plt.savefig('saved_plots/results_{}.png'.format(trainer.agent.model_id))

trainer.set_trained_dqn('saved_weights/target_solved_64_64_score_13.12.pth')
scores = []
for _ in range(100):
    trainer.play()

trainer.env.close()
