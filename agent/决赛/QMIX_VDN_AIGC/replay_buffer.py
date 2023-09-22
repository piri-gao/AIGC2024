import numpy as np
import torch
import copy


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.device = args.device
        self.episode_num = 0
        self.current_size = 0
        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim]),
                       's': np.zeros([self.buffer_size, self.episode_limit + 1, self.state_dim]),
                       'avail_a_n': np.ones([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'last_onehot_a_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'fixed_a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'dw': np.ones([self.buffer_size, self.episode_limit, 1]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'active_each': np.zeros([self.buffer_size, self.episode_limit, self.N])
                       }
        self.episode_len = np.zeros(self.buffer_size)
    
    def add_episode_replay(self, replay_episodes):
        for i in range(len(replay_episodes)):
            replay_episode = replay_episodes[i]
            for j in range(len(replay_episode)):
                if j != len(replay_episode)-1:
                    episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, fixed_a_n, r, done, dw = replay_episode[j]
                    self.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, fixed_a_n, r, done, dw)
                else:
                    episode_step, obs_n, s, avail_a_n = replay_episode[j]
                    self.store_last_step(episode_step, obs_n, s, avail_a_n)

    def store_transition(self, episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, fixed_a_n, r, done, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step + 1] = last_onehot_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['fixed_a_n'][self.episode_num][episode_step] = fixed_a_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw
        self.buffer['active'][self.episode_num][episode_step] = 1.0
        self.buffer['active_each'][self.episode_num][episode_step] = 1 - np.array(done)

    def store_last_step(self, episode_step, obs_n, s, avail_a_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}
        for key in self.buffer.keys():
            if key == 'obs_n' or key == 's' or key == 'avail_a_n' or key == 'last_onehot_a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32).to(self.device)
            elif key == 'a_n' or key == 'fixed_a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.long).to(self.device)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32).to(self.device)

        return batch, max_episode_len
