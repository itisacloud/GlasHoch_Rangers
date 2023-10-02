import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim, nn
from torch.cuda import device

from PolicyNetwork import PolicyNetwork
from State import State
from expert import Expert

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', "BOMB"]

reversed = {"UP": 0,
            "RIGHT": 1,
            "DOWN": 2,
            "LEFT": 3,
            "WAIT": 4,
            "BOMB": 5
            }


class Agent:
    def __init__(self, AGENT_CONFIG, REWARD_CONFIG, training):
        # Define a policy network
        self.training = training

        self.agent_name = AGENT_CONFIG["agent_name"]
        self.config_name = AGENT_CONFIG["config_name"]

        self.exploration_rate_min = AGENT_CONFIG["exploration_rate_min"]
        self.exploration_rate_decay = AGENT_CONFIG["exploration_rate_decay"]
        self.exploration_rate = AGENT_CONFIG["exploration_rate"]

        self.imitation_learning = AGENT_CONFIG["imitation_learning"]  # if true, the agent will learn from an expert
        self.imitation_learning_rate = AGENT_CONFIG["imitation_learning_rate"]
        self.imitation_learning_decay = AGENT_CONFIG["imitation_learning_decay"]
        self.imitation_learning_min = AGENT_CONFIG["imitation_learning_min"]
        self.imitation_learning_expert = AGENT_CONFIG["imitation_learning_expert"]

        if self.imitation_learning:
            self.imitation_learning_expert_name = self.imitation_learning_expert
            self.imitation_learning_expert = Expert(self.imitation_learning_expert)
        else:
            self.imitation_learning_expert_name = "no_expert"

        self.state_dim = AGENT_CONFIG["state_dim"]
        self.action_dim = AGENT_CONFIG["action_dim"]
        self.batch_size = AGENT_CONFIG["batch_size"]
        self.save_every = AGENT_CONFIG["save_every"]
        self.curr_step = 0
        self.since_last_save = 0  # fall back for the save_every

        # Hyperparameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.save_dir = "./models"

        # setting up the network
        self.policy = PolicyNetwork().to(self.device)

        self.gamma = AGENT_CONFIG["gamma"]
        self.lamb = AGENT_CONFIG["lambda"]
        self.worker_steps = AGENT_CONFIG["worker_steps"]
        self.n_mini_batch = AGENT_CONFIG["n_mini_batch"]
        self.epochs = AGENT_CONFIG["epochs"]
        self.lr = AGENT_CONFIG["learning_rate"]
        self.exploration_method = AGENT_CONFIG["exploration_method"]

        self.rewards = []
        self.mini_batch_size = self.batch_size // self.n_mini_batch

        # optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr},
            {'params': self.policy.critic.parameters(), 'lr': self.lr}  # do we need different lrs?
        ], eps=1e-4)

        # implement lr scheduler later
        if AGENT_CONFIG["lr_scheduler"]:
            self.lr_scheduling = True
            self.lr_scheduler_step = AGENT_CONFIG["lr_scheduler_step"]
            self.lr_scheduler_gamma = AGENT_CONFIG["lr_scheduler_gamma"]
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.policy.optimizer, step_size=self.lr_scheduler_step,
                                                                gamma=self.lr_scheduler_gamma)
            self.lr_scheduler_min = AGENT_CONFIG["lr_scheduler_min"]

        # discount factor
        self.mse_loss = nn.MSELoss()

        self.policy_old = PolicyNetwork().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.all_episode_rewards = []
        self.all_mean_rewards = []
        self.episode = 0

        self.REWARD_CONFIG = REWARD_CONFIG

        if AGENT_CONFIG["load"]:  # load model :D
            self.load_checkpoint(AGENT_CONFIG["load_path"])

    def act(self, features, state=None):
        if self.exploration_method == "epsilon-greedy" and self.training == True:
            if np.random.rand() < self.exploration_rate:
                if np.random.rand() < self.imitation_learning_rate and self.imitation_learning:
                    try:
                        action_idx = reversed[self.imitation_learning_expert.act(state)]
                    except:
                        action_idx = np.random.randint(self.action_dim)
                else:
                    action_idx = np.random.randint(self.action_dim)
            else:
                action_values = self.policy(features, model="online")
                action_idx = torch.argmax(action_values, axis=1).item()
        elif self.training:
            raise ValueError("exploration_method must be epsilon-greedy or boltzmann")
        else:
            action_values = self.policy(features, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.imitation_learning_rate *= self.imitation_learning_decay
        self.imitation_learning_rate = max(self.imitation_learning_rate, self.imitation_learning_min)
        if self.lr_scheduling:
            self.lr_scheduler.step()
            for param_group in self.policy.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], self.lr_scheduler_min)

        self.curr_step += 1
        return action_idx

    def save_checkpoint(self):
        filename = os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(self.episode))
        torch.save(self.policy_old.state_dict(), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    def load_checkpoint(self, filename):
        self.policy.load_state_dict(torch.load(os.path.join(self.save_dir, filename)))
        self.policy_old.load_state_dict(torch.load(os.path.join(self.save_dir, filename)))
        print('Resuming training from checkpoint \'{}\'.'.format(filename))

    def calculate_loss(self, samples, clip_range):
        sampled_returns = samples['returns']
        sampled_advantages = samples['advantages']
        pi, value = self.policy(samples['obs'])
        ratio = torch.exp(pi.log_prob(samples['actions']) - samples['log_pis'])
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages)
        entropy_bonus = pi.entropy()
        vf_loss = self.mse_loss(value, sampled_returns)
        loss = -policy_reward + 0.5 * vf_loss - 0.01 * entropy_bonus
        return loss.mean()

    def train(self, memory, clip_range):
        samples = memory.get_samples()
        indexes = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mini_batch_indexes = indexes[start: end]
            mini_batch = {}
            for k, v in samples.items():
                mini_batch[k] = v[mini_batch_indexes]
            for _ in range(self.epochs):
                loss = self.calculate_loss(clip_range=clip_range, samples=mini_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())

    def sample(self):
        rewards = np.zeros(self.worker_steps, dtype=np.float32)
        actions = np.zeros(self.worker_steps, dtype=np.int32)
        done = np.zeros(self.worker_steps, dtype=bool)
        obs = np.zeros((self.worker_steps, 4, 84, 84), dtype=np.float32)
        log_pis = np.zeros(self.worker_steps, dtype=np.float32)
        values = np.zeros(self.worker_steps, dtype=np.float32)
        for t in range(self.worker_steps):
            with torch.no_grad():
                obs[t] = self.obs
                pi, v = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
                values[t] = v.cpu().numpy()
                a = pi.sample()
                actions[t] = a.cpu().numpy()
                log_pis[t] = pi.log_prob(a).cpu().numpy()
            self.obs, rewards[t], done[t], _ = env.step(actions[t])
            self.obs = self.obs.__array__()
            env.render()
            self.rewards.append(rewards[t])
            if done[t]:
                self.episode += 1
                self.all_episode_rewards.append(np.sum(self.rewards))
                self.rewards = []
                env.reset()
                if self.episode % 10 == 0:
                    print('Episode: {}, average reward: {}'.format(self.episode,
                                                                   np.mean(self.all_episode_rewards[-10:])))
                    self.all_mean_rewards.append(np.mean(self.all_episode_rewards[-10:]))
                    plt.plot(self.all_mean_rewards)
                    plt.savefig("{}/mean_reward_{}.png".format(self.save_directory, self.episode))
                    plt.clf()
                    self.save_checkpoint()
        returns, advantages = self.calculate_advantages(done, rewards, values)
        return {
            'obs': torch.tensor(obs.reshape(obs.shape[0], *obs.shape[1:]), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_pis': torch.tensor(log_pis, device=device),
            'advantages': torch.tensor(advantages, device=device, dtype=torch.float32),
            'returns': torch.tensor(returns, device=device, dtype=torch.float32)
        }
