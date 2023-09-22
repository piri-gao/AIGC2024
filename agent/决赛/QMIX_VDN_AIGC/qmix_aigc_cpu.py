import torch
import torch.nn as nn
import numpy as np
from agent.QMIX_VDN_AIGC.qmix_aigc import Q_network_RNN,Q_network_MLP



class QMIX_AIGC_CPU(object):
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id
        self.use_rnn = args.use_rnn

        # Compute the input dimension
        self.input_dim = self.obs_dim
        if self.add_last_action:
            print("------add last action------")
            self.input_dim += self.action_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.input_dim += self.N

        if self.use_rnn:
            print("------use RNN------")
            self.eval_Q_net_cpu = Q_network_RNN(args, self.input_dim)
        else:
            print("------use MLP------")
            self.eval_Q_net_cpu = Q_network_MLP(args, self.input_dim)
        self.train_step = 0

    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
            else:
                inputs = []
                obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
                inputs.append(obs_n)
                if self.add_last_action:
                    last_a_n = torch.tensor(last_onehot_a_n, dtype=torch.float32)
                    inputs.append(last_a_n)
                if self.add_agent_id:
                    inputs.append(torch.eye(self.N))

                inputs = torch.cat([x for x in inputs], dim=-1) # inputs.shape=(N,inputs_dim)
                q_value = self.eval_Q_net_cpu(inputs)

                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
                q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions
                a_n = q_value.argmax(dim=-1).numpy()
            return a_n