import argparse
import logging
import os
import random
import matplotlib.pyplot as plt

import gym
import numpy as np
import torch


import gc
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# from wrappers.normalized_actions import NormalizedActions

from collections import namedtuple


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="BipedalWalker-v3",
                    help="the environment on which the agent should be trained "
                         "(Default: RoboschoolInvertedPendulumSwingup-v1)")
parser.add_argument("--render_train", default=False, type=bool,
                    help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=False, type=bool,
                    help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default="./saved_models/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed (default: 0)")
parser.add_argument("--timesteps", default=1e6, type=int,
                    help="Num. of total timesteps of training (default: 1e6)")
parser.add_argument("--batch_size", default=64, type=int,
                    help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e6, type=int,
                    help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.2, type=int,
                    help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=2, default=[400, 300], type=tuple,
                    help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=10, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
args = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output Layer
        self.mu = nn.Linear(hidden_size[1], num_outputs)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        mu = torch.tanh(self.mu(x))
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also 
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # Define the actor
        self.actor = Actor(hidden_size, num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size, num_inputs, self.action_space).to(device)

        # Define the critic
        self.critic = Critic(hidden_size, num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size, num_inputs, self.action_space).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=1e-4)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-2
                                     )  # optimizer for the critic network

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def calc_action(self, state, action_noise=None):
        x = state.to(device)

        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])

        return mu

    def update_params(self, batch):
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


if __name__ == "__main__":
    env = gym.make(args.env)
    # env = NormalizedActions(env)

    # Set random seed for all used libraries where possible
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)
    agent = DDPG(args.gamma,
                 args.tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 )

    # Initialize replay memory
    memory = ReplayMemory(int(args.replay_size))

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(args.noise_stddev) * np.ones(nb_actions))

    # Define counters and other variables
    start_step = 0
    # timestep = start_step
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0

    while timestep <= args.timesteps:
        ou_noise.reset()
        epoch_return = 0

        state = torch.Tensor([env.reset()]).to(device)
        while True:
            if args.render_train:
                env.render()

            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                break

        print("Len rewards: ", len(rewards))
        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)

        print(f"Episode {len(rewards)}, timestep {timestep}, last reward: {epoch_return}")

        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if timestep >= 10000 * t:
            t += 1
            test_rewards = []
            for _ in range(args.n_test_cycles):
                state = torch.Tensor([env.reset()]).to(device)
                test_reward = 0
                while True:
                    if args.render_eval:
                        env.render()

                    action = agent.calc_action(state)  # Selection without noise

                    next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                    test_reward += reward

                    next_state = torch.Tensor([next_state]).to(device)

                    state = next_state
                    if done:
                        break
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

        epoch += 1

    env.close()

    plt.plot(rewards)
    plt.savefig("ddpg_rewards.png")

    print("Succes.")
