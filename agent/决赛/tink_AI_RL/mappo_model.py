import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from torch.distributions import Categorical

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class GRUActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GRUActor, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim*10)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden = None

    def init_hidden(self):
        self.hidden = None

    def forward(self, state, actions_mask=None):
        batch_size = state.shape[0]
        gru_out, self.hidden = self.gru(state, self.hidden)
        logits = self.fc(gru_out)
        logits = logits.reshape(batch_size,10,-1)
        if actions_mask!=None:
            logits[actions_mask==0] = -99999999
        probs = self.softmax(logits)
        return probs


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 10)
        self.hidden = None

    def init_hidden(self):
        self.hidden = None

    def forward(self, state):
        batch_size = state.shape[0]
        gru_out, self.hidden = self.gru(state, self.hidden)
        value = self.fc(gru_out)
        value = value.reshape(batch_size,10,-1)
        return value

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MAPPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, buffer_size, batch_size):
        self.obs_mixer = MultiAgentTransformer(n_agents=10, input_dim=state_dim, hidden_dim=16, output_dim=128).to(device)
        self.actor = GRUActor(1330, action_dim, hidden_dim).to(device)
        self.critic = Critic(1330, hidden_dim).to(device)
        self.params = list(self.obs_mixer.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = optim.Adam(self.params, lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.init_hidden()
    
    def init_hidden(self):
        self.actor.init_hidden()
        self.critic.init_hidden()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        state = self.obs_mixer(state)
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        return action
    
    def evaluate(self, state, global_obs, actions_mask):
        actions_mask = torch.FloatTensor(actions_mask).unsqueeze(0).to(device)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        global_obs = torch.FloatTensor(global_obs).unsqueeze(0).to(device).expand(1,10,-1)
        state = self.obs_mixer(state).to(device)
        state = torch.cat([state, global_obs],dim=-1).reshape(1,-1)
        probs = self.actor(state, actions_mask).reshape(1,10,-1)
        m = Categorical(probs)
        action = m.sample()
        a_log_probs = m.log_prob(action)
        return action, a_log_probs

    def train(self):
        self.init_hidden()
        if len(self.replay_buffer) < self.batch_size:
            return None,None

        # Sample experiences from the replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        states, global_obs, actions, a_log_probs, rewards, next_states, global_next_obs, dones = zip(*experiences)
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(device)
        batch_size = states.shape[0]
        global_obs = torch.FloatTensor(global_obs).unsqueeze(1).expand(batch_size,10,-1).to(device)
        global_next_obs = torch.FloatTensor(global_next_obs).unsqueeze(1).expand(batch_size,10,-1).to(device)
        actions = torch.LongTensor(actions).to(device)
        a_log_probs = torch.FloatTensor(a_log_probs).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(2).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(2).to(device)
        
        states = self.obs_mixer(states)
        states = torch.cat([states,global_obs],dim=-1).reshape(batch_size,-1)
        next_states = self.obs_mixer(next_states)
        next_states = torch.cat([next_states, global_next_obs],dim=-1).reshape(batch_size,-1)
        
        # Critic loss
        predicted_values = self.critic(states)
        target_values = self.critic(next_states)
        target_values = rewards + self.gamma * target_values * (1 - dones)
        critic_loss = nn.MSELoss()(predicted_values, target_values.detach())
        
        # Actor loss
        probs = self.actor(states)
        
        m = Categorical(probs)
        log_probs = m.log_prob(actions).reshape(batch_size,10,1)
        advantages = target_values - predicted_values.detach()
        old_log_probs = a_log_probs.reshape(batch_size,10,1).detach()
        ratio = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() + critic_loss
        a_loss = actor_loss.cpu().detach()
        # Update actor and critic
        self.ac_optimizer.zero_grad()
        actor_loss.backward()
        self.ac_optimizer.step()
        return a_loss,critic_loss.cpu().detach()

# def main():
#     env = gym.make('YourEnvironment-v0')
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     mappo = MAPPO(state_dim=state_dim, action_dim=action_dim, hidden_dim=64, lr=1e-4, gamma=0.99, epsilon=0.2, buffer_size=10000, batch_size=64)

#     num_episodes = 1000

#     for episode in range(num_episodes):
#         state = env.reset()
#         mappo.init_hidden()
#         done = False
#         while not done:
#             action = mappo.select_action(state)
#             next_state, reward, done, _ = env.step(action)
#             mappo.replay_buffer.add((state, action, reward, next_state, done))
#             state = next_state
#             mappo.train()

#         # Evaluate the agent periodically
#         if episode % 10 == 0:
#             avg_reward = np.mean([mappo.inference(env) for _ in range(10)])
#             print(f"Episode: {episode}, Average Reward: {avg_reward}")

# if __name__ == '__main__':
#     main()