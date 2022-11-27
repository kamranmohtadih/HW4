import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.mdp
import matplotlib.pyplot as plt
import hiive.mdptoolbox.example
import numpy as np
import sys
import time

# Global parameters
max_iter = 20000000
gamma = 0.8
epsilon = 0.9
epsilon_min = 0.4

num_states = 1000


def value_iteration(P, R):
    np.random.seed(666)
    value_iter = mdptoolbox.mdp.ValueIteration(P, R, gamma=gamma, max_iter=max_iter)
    value_iter.run()
    return value_iter


def policy_iteration(P, R, ite):
    np.random.seed(666)
    policy_iter = mdptoolbox.mdp.PolicyIteration(P, R, gamma=gamma, max_iter=ite)
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
P, R = mdptoolbox.example.forest(S=num_states, p=0.1)

q = []
q_x = []
v = []
p = []

q_t = []
v_t = []
p_t = []
v_x =[]

vi = value_iteration(P, R)
for i in vi.run_stats:
    v.append(i['Max V'])
    v_t.append(i['Time'])
    v_x.append((i['Iteration']))
print(vi.run_stats[-1])
print("Value iteration: ", vi.policy)

ql = Qlearning(P, R)
for i in ql.run_stats:
    q.append(i['Max V'])
    q_t.append(i['Time'])
    q_x.append((i['Iteration']))
print("Q-learning: ", ql.policy)

pi = policy_iteration(P, R, 100)
p_x =[]
for i in pi.run_stats:
    p.append(round(i['Max V']))
    p_t.append(i['Time'])
    p_x.append(i['Iteration'])
print(pi.run_stats[-1])

print("policy iteration: ", pi.policy)
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(p_x, p, 'b--', label='Policy iteration - Max Value')
ax2.plot(p_x, p_t, 'b--', label='Policy iteration - Time')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Max Value')
ax1.set_title('Policy Iteration')
ax1.legend()
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Time')
ax2.set_title('Policy Iteration')
ax2.legend()
plt.savefig('forest/Policy_Iteration_Forest '+ str(num_states) +'.png')
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
plt.savefig('forest/Value_Iteration_Forest+  '+ str(num_states) +'.png')
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
plt.savefig('forest/QLearning_Forest' + str(epsilon) + '_' + str(max_iter) + '_' + str(epsilon_min) +  '_num_states_' + str(num_states) +'_'+ str(epsilon_min) +'.png')
plt.clf()
