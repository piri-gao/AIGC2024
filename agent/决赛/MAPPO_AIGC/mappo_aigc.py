import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import copy
import os
from torch.utils.data import BatchSampler,SequentialSampler


class MultiAgentTransformer(nn.Module):
    def __init__(self, n_agents, input_dim, hidden_dim, output_dim, n_layers=1, n_heads=8, dropout=0.1):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Transformer layers
        # 定义编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        # inputs: (batch_size, n_agents, input_dim)
        if len(inputs.size()) == 2:
            batch_size = 1
            n_agents, input_dim = inputs.size()
        else:
            batch_size, n_agents, input_dim = inputs.size()
        
        # Reshape inputs to be (batch_size * n_agents, input_dim)
        inputs = inputs.view(batch_size * n_agents, input_dim)
        # Embed inputs
        embedded = self.embedding(inputs)
        # Reshape embedded to be (n_agents, batch_size, hidden_dim)
        embedded = embedded.view(batch_size, n_agents, -1).transpose(0, 1)
        # Use transformer to process inputs (seq_len, batch_size, input_dim)
        outputs = self.transformer_encoder(embedded)
        # Reshape outputs to be (batch_size * n_agents, hidden_dim)
        outputs = outputs.transpose(0, 1).contiguous().view(batch_size * n_agents, -1)
        # Use output layer to generate output vectors
        outputs = self.output(outputs)
        # Reshape outputs to be (batch_size, n_agents, output_dim)
        
        if batch_size==1:
            outputs = outputs.view(n_agents, -1)
        else:
            outputs = outputs.view(batch_size, n_agents, -1)
        return outputs
    
# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None
        self.obs_mixer = MultiAgentTransformer(n_agents=args.N, input_dim=actor_input_dim, hidden_dim=16, output_dim=128)
        self.fc1 = nn.Linear(128, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        x = self.obs_mixer(actor_input)
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(x))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        x = self.fc2(self.rnn_hidden)
        x[avail_a_n == 0] = -1e10  # Mask the unavailable actions
        prob = torch.softmax(x, dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, max_episode_len, N, actor_input_dim), prob.shape(mini_batch_size, max_episode_len, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        x = self.fc3(x)
        x[avail_a_n == 0] = -1e10  # Mask the unavailable actions
        prob = torch.softmax(x, dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, max_episode_len, N, critic_input_dim), value.shape=(mini_batch_size, max_episode_len, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO_AIGC:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.device = args.device
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
            self.actor = Actor_RNN(args, self.actor_input_dim).to(self.device)
            self.critic = Critic_RNN(args, self.critic_input_dim).to(self.device)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim).to(self.device)
            self.critic = Critic_MLP(args, self.critic_input_dim).to(self.device)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

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
            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1).to(self.device)  # actor_input.shape=(N, actor_input_dim)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32).to(self.device)  # avail_a_n.shape=(N, action_dim)
            prob = self.actor(actor_inputs, avail_a_n).to('cpu') # prob.shape=(N, action_dim)
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
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1).to(self.device) # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # Get training data
        max_episode_len = replay_buffer.max_episode_len

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len,N)
            deltas = batch['r'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['dw']) - batch['v_n'][:, :-1]
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,max_episode_len,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,max_episode_len,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv_copy = copy.deepcopy(adv.cpu().numpy())
                adv_copy[batch['active'].cpu().numpy() == 0] = np.nan
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch, max_episode_len)
        a_l = 0
        c_l = 0
        bl_l = 0
        count = 0
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    Get probs_now and values_now
                    probs_now.shape=(mini_batch_size, max_episode_len, N, action_dim)
                    values_now.shape=(mini_batch_size, max_episode_len, N)
                """
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(max_episode_len):
                        # prob.shape=(mini_batch_size*N, action_dim)
                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1),
                                          batch['avail_a_n'][index, t].reshape(self.mini_batch_size * self.N, -1))
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))  # v.shape=(mini_batch_size*N,1)
                        values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index], batch['avail_a_n'][index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)
                a_rule = batch['fixed_a_n'][index]
                try:
                    dist_now = Categorical(probs_now)
                    dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, max_episode_len, N)
                except:
                    import pdb;pdb.set_trace()
                # batch['a_n'][index].shape=(mini_batch_size, max_episode_len, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, max_episode_len, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())  # ratios.shape=(mini_batch_size, max_episode_len, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                bl_loss = 0
                if self.use_bl:
                    bl_loss = torch.nn.CrossEntropyLoss(reduction = 'none')(probs_now.reshape(-1, self.action_dim), a_rule.reshape(-1))
                    bl_loss = (bl_loss.reshape(self.mini_batch_size,-1,self.N) * batch['active'][index]).sum() / batch['active'][index].sum()
                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss + bl_loss
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()
                a_l += actor_loss.mean().item()
                c_l += critic_loss.mean().item()
                bl_l += bl_loss.mean().item()
                count += 1
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        return a_l/count, c_l/count, bl_l/count

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.use_agent_specific:
            critic_inputs.append(batch['obs_n'])
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len, 1, 1).to(self.device)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        state = {
                'total_steps': total_steps,
                'eva_rewards': seed,
                'opt_state_dict': self.ac_optimizer.state_dict(),
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                 }
        filename = "agent/MAPPO_AIGC/model/MAPPO_env_{}_actor_number_{}_return_{}_step_{}k.pth".format(env_name, number, int(seed), int(total_steps / 1000))
        torch.save(state, filename)

        print("已保存：{}".format(filename))

    def load_model(self, env_name, number, seed, step):
        filename = "agent/MAPPO_AIGC/model/MAPPO_env_{}_actor_number_{}_return_{}_step_{}k.pth".format(env_name, number, int(seed), step)
        total_steps = 0
        best_eva_rewards = 0
        if not os.path.exists(filename):
            print("没有该模型,请修改reward参数值")
            return total_steps, best_eva_rewards
        try:
            state = torch.load(filename)
        except:
            state = torch.load(filename, map_location=lambda storage, loc:storage)
        print("已加载：{}".format(filename))
        total_steps = state['total_steps']
        best_eva_rewards = state['eva_rewards']
        self.actor.load_state_dict(state['actor_state_dict'])
        self.critic.load_state_dict(state['critic_state_dict'])
        self.ac_optimizer.load_state_dict(state['opt_state_dict'])
        print('预训练的学习率 = {}'.format(self.ac_optimizer.param_groups[0]['lr']))
        return total_steps, best_eva_rewards
