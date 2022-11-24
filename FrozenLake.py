import gym
import hiive.mdptoolbox as mdptoolbox
from MDP import MarkovDecisionProcess as MDP
import hiive.mdptoolbox.mdp
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import pandas as pd

# Global parameters
myReward = 0
finalReward = 10
holeReward = -10
max_iter = 100000
gamma = 0.99
epsilon = 0.9
epsilon_min = 0.4


def value_iteration(P, R):
    np.random.seed(666)
    value_iter = mdptoolbox.mdp.ValueIteration(P, R, gamma=gamma, max_iter=max_iter)
    value_iter.run()
    return value_iter


def policy_iteration(P, R, iter):
    np.random.seed(666)
    policy_iter = mdptoolbox.mdp.PolicyIteration(P, R, gamma=gamma, max_iter=iter)
    policy_iter.run()
    return policy_iter


def Qlearning(P, R):
    np.random.seed(666)
    qlearner = mdptoolbox.mdp.QLearning(P, R, gamma=gamma, n_iter=max_iter, epsilon=epsilon, epsilon_decay=0.1,
                                        epsilon_min=epsilon_min)
    qlearner.setVerbose()
    qlearner.run()
    return qlearner


np.set_printoptions(threshold=sys.maxsize)

env = gym.make('FrozenLake8x8-v1', is_slippery=True, render_mode="rgb_array")
env.reset()
plt.imshow(env.render())
plt.show()
mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)

print("Number of states ", mdp.num_states)
print("Number of actions ", mdp.num_actions)

all_actions = {'LEFT': 0, 'DOWN': 1, 'RIGHT': 2, 'UP': 3}
all_states = range(0, 64)

P = np.zeros((mdp.num_actions, mdp.num_states, mdp.num_states))
R = np.zeros((mdp.num_states, mdp.num_actions))
print("======================================")


for s in all_states:
    for a in all_actions.keys():
        for i in range(len(mdp.P[s][all_actions[a]])):
            tran_prob = mdp.P[s][all_actions[a]][i][0]
            state_ = mdp.P[s][all_actions[a]][i][1]
            R[s][all_actions[a]] += tran_prob * mdp.P[s][all_actions[a]][i][2]
            P[all_actions[a], s, state_] += tran_prob


q = []
q_x = []
v = []
p = []

q_t = []
v_t = []
p_t = []

vi = value_iteration(P, R)
print(vi.run_stats[-1])
for i in vi.run_stats:
    v.append(i['Max V'])
    v_t.append(i['Time'])

ql = Qlearning(P, R)
for i in ql.run_stats:
    q.append(i['Max V'])
    q_t.append(i['Time'])
    q_x.append((i['Iteration']))

pi = policy_iteration(P, R, 10)
for i in pi.run_stats:
    p.append(i['Max V'])
    p_t.append(i['Time'])

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(range(1, 11), p, 'b--', label='Policy iteration - Max Value')
ax2.plot(range(1, 11), p_t, 'b--', label='Policy iteration - Time')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Max Value')
ax1.set_title('Policy Iteration')
ax1.legend()
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Time')
ax2.set_title('Policy Iteration')
ax2.legend()
plt.savefig('images/Policy_Iteration_Frozenlake.png')
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(v, 'b--', label='Value iteration - Max Value')
ax2.plot(v_t, 'b--', label='Value iteration - Time')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Max Value')
ax1.set_title('Value Iteration')
ax1.legend()
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Time')
ax2.set_title('Value Iteration')
ax2.legend()
plt.savefig('images/Value_Iteration_Frozenlake.png')
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(q_x, q, 'b--', label='QLearning - Max Value')
ax2.plot(q_x, q_t, 'b--', label='QLearning - Time')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Max Value')
ax1.set_title('QLearning')
ax1.legend()
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Time')
ax2.set_title('QLearning')
ax2.legend()
plt.savefig('images/QLearning_Frozenlake' + str(epsilon) + '_' + str(max_iter) + '_' + str(epsilon_min) + '.png')
plt.clf()
