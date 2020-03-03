import numpy as np
from collections import namedtuple
import random
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

Transition = namedtuple("Transition", ("state", "action", "done",
                                       "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


device = "cuda:0"
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


class OUNoise:
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt
        dx += self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state


def fan_in_uniform_init(tensor, fan_in=None):
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        self.mu = nn.Linear(hidden_size[1], num_outputs)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT,
                         WEIGHTS_FINAL_INIT)
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

        mu = torch.tan(self.mu(x))
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
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer
        self.V = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT,
                         WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), 1)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        v = self.V(x)
        return v


def soft_update(target, source, tau):
    for target_p, p in zip(target.parameters(), source.parameters()):
        target_p.data.copy_(target_p.data * (1.0 - tau) + p.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG:
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        self.actor = Actor(hidden_size, num_inputs, self.action_space)
        self.actor.to(device)
        self.actor_target = Actor(hidden_size, num_inputs, self.action_space)
        self.actor_target.to(device)

        self.critic = Critic(hidden_size, num_inputs, self.action_space)
        self.critic.to(device)
        self.critic_target = Critic(hidden_size, num_inputs, self.action_space)
        self.critic_target.to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-3)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        x = state.to(device)

        self.actor.eval()
        mu = self.actor(x)
        self.actor.train()

        if action_noise is not None:
            noise = torch.from_numpy(action_noise.noise()).float().to(device)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])
        return mu

    def update_params(self, batch):
        # Get tensors from the batch
        state_batch = torch.cat(batch.state, 0).float().to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state, 0).to(device)

        # Compute actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch,
                                                      next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqeeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * \
                self.gamma * next_state_action_values

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
    env = gym.make("LunarLanderContinuous-v2")

    batch_size = 64

    gamma = 0.99
    tau = 0.001
    hidden_size = [128, 128]
    agent = DDPG(gamma, tau, hidden_size, env.observation_space.shape[0],
                 env.action_space)

    memory = ReplayMemory(1_000_000)

    nb_actions = env.action_space.shape[0]
    ou_noise = OUNoise(nb_actions)

    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0

    timestep = 0
    while timestep <= 100_000:
        ou_noise.reset()
        epoch_return = 0

        state = torch.Tensor([env.reset()]).float().to(device)
        while True:
            action = agent.select_action(state, ou_noise).detach()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).float().to(device)
            reward = torch.Tensor([reward]).float().to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)

        test_rewards = []
        if timestep > 10_000 * t:
            t += 1
            for _ in range(10):
                state = torch.from_numpy(env.reset()).to(device)
                test_reward = 0
                while True:
                    action = agent.select_action(state).cpu().numpy().item()
                    next_state, reward, done, _ = env.step(action)
                    test_reward += reward

                    next_state = torch.from_numpy(next_state).to(device)
                    state = next_state
                    if done:
                        break

                test_rewards.append(np.mean(test_reward))

        print("Success.")
        print("Rewards, losses: ", len(rewards), len(value_losses), len(policy_losses))
        print("Test rewards: ", len(test_rewards))
