import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .episode_buffer import EpisodeBatch

class MultiAgentTransformer(nn.Module):
    def __init__(self, n_agents, input_dim, hidden_dim, output_dim, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers, dropout=dropout)

        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        # inputs: (batch_size, n_agents, input_dim)
        batch_size, n_agents, input_dim = inputs.size()

        # Reshape inputs to be (batch_size * n_agents, input_dim)
        inputs = inputs.view(batch_size * n_agents, input_dim)

        # Embed inputs
        embedded = self.embedding(inputs)

        # Reshape embedded to be (n_agents, batch_size, hidden_dim)
        embedded = embedded.view(batch_size, n_agents, -1).transpose(0, 1)

        # Use transformer to process inputs
        outputs = self.transformer(embedded, embedded)

        # Reshape outputs to be (batch_size * n_agents, hidden_dim)
        outputs = outputs.transpose(0, 1).contiguous().view(batch_size * n_agents, -1)

        # Use output layer to generate output vectors
        outputs = self.output(outputs)

        # Reshape outputs to be (batch_size, n_agents, output_dim)
        outputs = outputs.view(batch_size, n_agents, -1)

        return outputs


class GRUQNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=64):
        super(GRUQNetwork, self).__init__()
        self.gru = nn.GRU(input_shape, hidden_size)
        self.fc = nn.Linear(hidden_size, n_actions)
        self.hx = None

    def forward(self, x):
        x, self.hx= self.gru(x, self.hx)
        q = self.fc(x)
        return q
    def init_gru(self):
        self.hx = None
    
class MixingNetwork(nn.Module):
    def __init__(self, state_shape, n_agents, hidden_size=64):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.embed_dim = hidden_size
        self.abs = True
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        hypernet_embed = hidden_size
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        
        return q_tot

class QMIX:
    def __init__(self, n_agents, state_shape, obs_shape, n_actions, gamma=0.99, lr=1e-3, tau=0.01):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.obs_generator = MultiAgentTransformer(n_agents, obs_shape, hidden_dim=16, output_dim=64)
        obs_shape = 64
        self.q_networks = [GRUQNetwork(obs_shape, n_actions) for _ in range(n_agents)]
        self.target_q_networks = [GRUQNetwork(obs_shape, n_actions) for _ in range(n_agents)]
        self.mixing_network = MixingNetwork(state_shape, n_agents)
        self.target_mixing_network = MixingNetwork(state_shape, n_agents)
        self.q_parameters = None
        self.obs_generator_optimizer = optim.Adam(self.obs_generator.parameters(), lr=lr)
        self.q_optimizers = [optim.Adam(q_net.parameters(), lr=lr) for q_net in self.q_networks]
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=lr)
        self.hard_update()
    
    def init_hidden(self):
        for i in range(self.n_agents):
            self.q_networks[i].init_gru()

    def evaluate(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        actions = []
        obs = self.obs_generator(obs)
        for i in range(self.n_agents):
            action = self.q_networks[i](obs[:, i, :]).cpu().detach().numpy().squeeze(0)
            actions.append(action)
        return actions
    
    def hard_update(self):
        for i in range(self.n_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def soft_update(self):
        for i in range(self.n_agents):
            for target_param, param in zip(self.target_q_networks[i].parameters(), self.q_networks[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_mixing_network.parameters(), self.mixing_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def forward(self, ep_batch, t, test_mode=False):

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
    
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        inputs.append(batch['global'][:, t])
        if t == 0:
            inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
        else:
            inputs.append(batch["actions_onehot"][:, t-1])
        inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        mac_out = []
        self.init_hidden()
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_back = chosen_action_qvals
        
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = torch.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()
        
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def train(self, batch):
        # batch is a tuple (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = batch

        q_preds = []
        q_targets = []

        for i in range(self.n_agents):
            q_pred = self.q_networks[i](states[:, i, :])
            q_preds.append(q_pred)
            q_target = self.target_q_networks[i](next_states[:, i, :])
            q_targets.append(q_target)

        q_preds = torch.stack(q_preds, dim=-1)
        q_targets = torch.stack(q_targets, dim=-1)

        q_total_pred = self.mixing_network(q_preds)
        q_total_target = self.target_mixing_network(q_targets)

        td_error = rewards - (self.gamma * q_total_target * (1 - dones))

        q_loss = (td_error ** 2).mean()
        mixing_loss = (q_total_pred - q_total_target.detach()).pow(2).mean()

        for i in range(self.n_agents):
            self.q_optimizers[i].zero_grad()
        self.mixing_optimizer.zero_grad()
        self.obs_generator_optimizer.zero_grad()

        total_loss = q_loss + mixing_loss
        total_loss.backward()

        for i in range(self.n_agents):
            self.q_optimizers[i].step()
        self.mixing_optimizer.step()
        self.obs_generator_optimizer.step()

        self.soft_update()