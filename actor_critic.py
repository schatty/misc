import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

device = "cuda:0"
SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values

    
model = Policy()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = 1e-9


def select_action(state):
    state = torch.from_numpy(state).float().to(device)
    probs, state_value = model(state)

    m = Categorical(probs)
    action = m.sample()

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []

    # calculate discounted rewards
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        r_target = torch.tensor([R]).to(device)
        value_losses.append(F.smooth_l1_loss(value, r_target))

    optimizer.zero_grad()

    # sum up all the values of policy losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()

    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10

    ep_rewards = []
    for i_episode in range(5_000):
        state = env.reset()
        ep_reward = 0

        for t in range(1, 10_000):
            action = select_action(state)

            state, reward, done, _ = env.step(action)

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        ep_rewards.append(ep_reward)

        # Update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        finish_episode()

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward))

    plt.plot(ep_rewards)
    plt.savefig("actor_critic_reward.png")


if __name__ == "__main__":
    main()
