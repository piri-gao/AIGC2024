import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import copy
from torch.utils.data import BatchSampler,SequentialSampler
from agent.MAPPO_AIGC.mappo_aigc import Actor_RNN,Critic_RNN,Actor_MLP,Critic_MLP

class MAPPO_AIGC_CPU:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.use_bl = args.use_bl
        self.add_agent_id = args.add_agent_id
        self.use_agent_specific = args.use_agent_specific
        self.use_value_clip = args.use_value_clip
        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N
        if self.use_agent_specific:
            print("------use agent specific global state------")
            self.critic_input_dim += args.obs_dim

        if self.use_rnn:
            print("------use rnn------")
            self.actor_cpu = Actor_RNN(args, self.actor_input_dim)
            self.critic_cpu = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor_cpu = Actor_MLP(args, self.actor_input_dim)
            self.critic_cpu = Critic_MLP(args, self.critic_input_dim)

    def choose_action(self, obs_n, avail_a_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3:
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                actor_inputs.append(torch.eye(self.N))
            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
            prob = self.actor_cpu(actor_inputs, avail_a_n) # prob.shape=(N, action_dim)
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                a_n = prob.argmax(dim=-1)
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.numpy(), a_logprob_n.numpy()
            
    def get_value(self, s, obs_n):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.use_agent_specific:  # Add local obs of agents
                critic_inputs.append(torch.tensor(obs_n, dtype=torch.float32))
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1) # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic_cpu(critic_inputs)  # v_n.shape(N,1)
            return v_n.cpu().numpy().flatten()