import sys
import torch
import gym
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


GAMMA = 0.99


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().to("cuda:0").unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions,
                                               p=np.squeeze(probs.detach().cpu().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw += 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards.to("cuda:0")
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


def main():
    env = gym.make("CartPole-v0")
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)
    policy_net.to("cuda:0")

    max_episode_num = 5_000
    max_steps = 10_000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        for step in range(max_steps):
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(step)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))

                if episode % 100 == 0:
                    print(f"Episode {episode}, reward {sum(rewards):.3f}")

                break

            state = new_state

    fig = plt.figure()
    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel("Episode")
    plt.savefig("reinforce_steps.png")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.savefig("reinforce_reward.png")
    plt.close(fig)

    print("Numsteps: ", numsteps)
    print("Reward: ", all_rewards)


if __name__ == "__main__":
    main()
