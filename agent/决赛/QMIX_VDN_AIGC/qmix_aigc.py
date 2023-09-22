import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent.QMIX_VDN_AIGC.mix_net import QMIX_Net, VDN_Net

class MultiLayerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(MultiLayerGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.gru_layers.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)
        if hx is None:
            hx = [torch.zeros(input.size(1), self.hidden_size, device=input.device) for i in range(self.num_layers)]
        else:
            hx = list(hx)

        output = []
        for i, gru_layer in enumerate(self.gru_layers):
            layer_output = []
            for seq in input:
                hx[i] = gru_layer(seq, hx[i])
                layer_output.append(hx[i])
            input = torch.stack(layer_output)
            output.append(input)

        if self.batch_first:
            output = [o.transpose(0, 1) for o in output]
        return output, hx
    
# orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Q_network_RNN(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size*N,input_dim)
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        Q = self.fc2(self.rnn_hidden)
        return Q


class Q_network_MLP(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_MLP, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size,max_episode_len,N,input_dim)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q


class QMIX_AIGC(object):
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.device = args.device
        self.gamma = args.gamma
        self.use_grad_clip = args.use_grad_clip
        self.batch_size = args.batch_size  # 这里的batch_size代表有多少个episode
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        self.use_rnn = args.use_rnn
        self.algorithm = args.algorithm
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_lr_decay = args.use_lr_decay
        self.use_bl = args.use_bl

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
            self.eval_Q_net = Q_network_RNN(args, self.input_dim).to(self.device)
            self.target_Q_net = Q_network_RNN(args, self.input_dim).to(self.device)
        else:
            print("------use MLP------")
            self.eval_Q_net = Q_network_MLP(args, self.input_dim).to(self.device)
            self.target_Q_net = Q_network_MLP(args, self.input_dim).to(self.device)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())

        if self.algorithm == "QMIX":
            print("------algorithm: QMIX------")
            self.eval_mix_net = QMIX_Net(args).to(self.device)
            self.target_mix_net = QMIX_Net(args).to(self.device)
        elif self.algorithm == "VDN":
            print("------algorithm: VDN------")
            self.eval_mix_net = VDN_Net().to(self.device)
            self.target_mix_net = VDN_Net().to(self.device)
        else:
            print("wrong!!!")
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters())
        if self.use_RMS:
            print("------optimizer: RMSprop------")
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        else:
            print("------optimizer: Adam------")
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)

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

                inputs = torch.cat([x for x in inputs], dim=-1).to(self.device)  # inputs.shape=(N,inputs_dim)
                q_value = self.eval_Q_net(inputs)

                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
                q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions
                a_n = q_value.argmax(dim=-1).numpy()
            return a_n.cpu()

    def train(self, replay_buffer, total_steps):
        batch, max_episode_len = replay_buffer.sample()  # Get training data
        self.train_step += 1
        a_l = 0
        count = 0

        inputs = self.get_inputs(batch, max_episode_len)  # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        if self.use_rnn:
            self.eval_Q_net.rnn_hidden = None
            self.target_Q_net.rnn_hidden = None
            q_evals, q_targets = [], []
            for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
                q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.input_dim))  # q_eval.shape=(batch_size*N,action_dim)
                q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.input_dim))
                q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))  # q_eval.shape=(batch_size,N,action_dim)
                q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            # Stack them according to the time (dim=1)
            q_evals = torch.stack(q_evals, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            q_evals = self.eval_Q_net(inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = self.target_Q_net(inputs[:, 1:])

        with torch.no_grad():
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, self.N, -1)
                q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
                q_evals_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999
                q_targets = q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len, N)

        # batch['a_n'].shape(batch_size,max_episode_len, N)
        q_values = q_evals
        q_evals = torch.gather(q_evals, dim=-1, index=batch['a_n'].unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N)
        a_rule = batch['fixed_a_n']
        # Compute q_total using QMIX or VDN, q_total.shape=(batch_size, max_episode_len, 1)
        if self.algorithm == "QMIX":
            q_total_eval = self.eval_mix_net(q_evals, batch['s'][:, :-1])
            q_total_target = self.target_mix_net(q_targets, batch['s'][:, 1:])
        else:
            q_total_eval = self.eval_mix_net(q_evals)
            q_total_target = self.target_mix_net(q_targets)
        # targets.shape=(batch_size,max_episode_len,1)
        targets = batch['r'] + self.gamma * (1 - batch['dw']) * q_total_target
        bl_loss = 0
        if self.use_bl:
            bl_loss = torch.nn.CrossEntropyLoss(reduction = 'none')(q_values.reshape(-1, self.action_dim), a_rule.reshape(-1))
            bl_loss = (bl_loss.reshape(self.batch_size,-1,self.N) * batch['active_each']).sum() / batch['active_each'].sum()
        td_error = (q_total_eval - targets.detach())
        mask_td_error = td_error * batch['active'] 
        loss = (mask_td_error ** 2).sum() / batch['active'].sum() + bl_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimizer.step()
        a_l += loss.mean().item()
        count += 1
        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.eval_Q_net.parameters(), self.target_Q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.eval_mix_net.parameters(), self.target_mix_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.use_lr_decay:
            self.lr_decay(total_steps)
        return a_l/count

    def lr_decay(self, total_steps):  # Learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        inputs = []
        inputs.append(batch['obs_n'])
        if self.add_last_action:
            inputs.append(batch['last_onehot_a_n'])
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len + 1, 1, 1).to(self.device)
            inputs.append(agent_id_one_hot)

        # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        inputs = torch.cat([x for x in inputs], dim=-1)

        return inputs

    def save_model(self, env_name, algorithm, number, seed, total_steps):
        filename = "agent/QMIX_VDN_AIGC/model/{}_{}_eval_rnn_number_{}_seed_{}_step_{}k.pth".format(env_name, algorithm, number, int(seed), int(total_steps / 1000))
        state = {
                'total_steps': total_steps,
                'eva_rewards': seed,
                'opt_state_dict': self.optimizer.state_dict(),
                'q_net_state_dict': self.eval_Q_net.state_dict(),
                'mix_net_state_dict': self.eval_mix_net.state_dict(),
                 }
        torch.save(state, filename)

    def load_model(self, env_name, algorithm, number, seed, total_steps):
        filename = "agent/QMIX_VDN_AIGC/model/{}_{}_eval_rnn_number_{}_seed_{}_step_{}k.pth".format(env_name, algorithm, number, int(seed), int(total_steps))
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
        self.eval_Q_net.load_state_dict(state['q_net_state_dict'])
        self.eval_mix_net.load_state_dict(state['mix_net_state_dict'])
        self.optimizer.load_state_dict(state['opt_state_dict'])
        print('预训练的学习率 = {}'.format(self.optimizer.param_groups[0]['lr']))
        return total_steps, best_eva_rewards
