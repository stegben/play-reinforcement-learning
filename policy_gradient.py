import sys

import gym
import numpy as np

import torch as T
import torch.optim as optim
from torch.autograd import Variable

from nn import MLP


class Policy(object):
    def __init__(self, input_dim, n_actions, gamma=0.9):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma

        self.model = MLP(input_dim, [32, 32], n_actions)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-2)

        self.action_reward = []

    def get_action(self, observation, stochastic=True):
        pred = self.model(observation)

        if stochastic:
            return pred.multinomial()
        return pred[0].argmax()

    def update(self):
        R = 0
        rewards = []
        for action, reward in self.action_reward:
            R = reward + self.gamma * R
            rewards.insert(0, R)

        rewards = T.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        actions = []
        for (action, _), reward in zip(self.action_reward, rewards):
            action.reinforce(reward)
            actions.append(action)

        self.optim.zero_grad()
        T.autograd.backward(actions, [None for _ in actions])
        self.optim.step()

        self.action_reward = []

    def record(self, action, reward):
        self.action_reward.append((action, reward))


def main():
    env = gym.make('CartPole-v0')
    n_actions = env.action_space.n
    input_dim = env.observation_space.shape[0]

    policy = Policy(input_dim, n_actions)

    reward = 0
    for n_episode in range(1000):
        if n_episode % 100 == 0:
            print('============== Episode {}'.format(n_episode))
            print('reward: {}'.format(reward))
        observation = env.reset()
        env.render()

        while True:
            if isinstance(observation, np.ndarray):
                observation = Variable(T.from_numpy(observation).float().unsqueeze(0))
            action = policy.get_action(observation)
            observation, reward, done, _ = env.step(action.data[0][0])
            env.render()
            policy.record(action, reward)

            if done:
                break

        policy.update()


if __name__ == '__main__':
    main()
