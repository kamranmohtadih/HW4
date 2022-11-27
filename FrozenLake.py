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
max_iter = 10000
gamma = 0.99
epsilon = 0.99
epsilon_min = 0.3


def show_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)

            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.savefig('images/'+title+str('.png'))
    plt.close()

    return plt

def colors():
    return {
        b'S': 'green',
        b'F': 'skyblue',
        b'H': 'black',
        b'G': 'gold',
    }

def directions():
    return {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }

def value_iteration(P, R, gamma):
    np.random.seed(666)
    value_iter = mdptoolbox.mdp.ValueIteration(P, R, gamma=gamma, max_iter=max_iter)
    value_iter.run()
    return value_iter


def policy_iteration(P, R, gamma, iter):
    np.random.seed(666)
    policy_iter = mdptoolbox.mdp.PolicyIteration(P, R, gamma=gamma, max_iter=iter)
    policy_iter.run()
    return policy_iter


def Qlearning(P, R, epsilon,epsilon_min,gamma):
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
p_x = []

vi = value_iteration(P, R,gamma)

plot = show_policy_map(
    'Optimum Policy_value_iter_Frozenlake' + str(epsilon) + '_' + str(max_iter) + '_' + str(epsilon_min)+ str(gamma) ,
    np.array(vi.policy).reshape(8,8), env.unwrapped.desc, colors(), directions())
for i in vi.run_stats:
    v.append(i['Max V'])
    v_t.append(i['Time'])



ql = Qlearning(P, R,epsilon,epsilon_min,gamma)
for i in ql.run_stats:
    q.append(i['Max V'])
    q_t.append(i['Time'])
    q_x.append((i['Iteration']))
plot = show_policy_map(
    'Optimum Policy_Qlearner_Frozenlake' + str(epsilon) + '_' + str(max_iter) + '_' + str(epsilon_min)+ str(gamma) ,
    np.array(ql.policy).reshape(8,8), env.unwrapped.desc, colors(), directions())



pi = policy_iteration(P, R, gamma,12)
plot = show_policy_map(
    'Optimum Policy_policy_iter_Frozenlake' + str(epsilon) + '_' + str(max_iter) + '_' + str(epsilon_min)+ str(gamma) ,
    np.array(pi.policy).reshape(8,8), env.unwrapped.desc, colors(), directions())
for i in pi.run_stats:
    p.append(i['Max V'])
    p_t.append(i['Time'])
    p_x.append((i['Iteration']))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot( p_x,p, 'b--', label='Policy iteration - Max Value')
ax2.plot( p_x, p_t, 'b--', label='Policy iteration - Time')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Max Value')
ax1.set_title('Policy Iteration')
ax1.legend()
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Time')
ax2.set_title('Policy Iteration')
ax2.legend()
plt.savefig('images/Policy_Iteration_Frozenlake'+ str(gamma) +'.png')
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
plt.savefig('images/Value_Iteration_Frozenlake'+ str(gamma) +'.png')
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
plt.savefig('images/QLearning_Frozenlake' + str(epsilon) + '_' + str(max_iter) + '_' + str(epsilon_min) + str(gamma) + '.png')
plt.clf()
