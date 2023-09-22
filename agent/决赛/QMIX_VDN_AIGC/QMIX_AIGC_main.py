import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from env.env_rl_runner import EnvRunner as AIGC2Env
from agent.QMIX_VDN_AIGC.replay_buffer import ReplayBuffer
from agent.QMIX_VDN_AIGC.qmix_aigc import QMIX_AIGC
from agent.QMIX_VDN_AIGC.qmix_aigc_cpu import QMIX_AIGC_CPU
from agent.QMIX_VDN_AIGC.normalization import Normalization, RewardScaling
from config import ADDRESS, config, ISHOST, XSIM_NUM
import multiprocessing as mp

def run_episode_aigc(evaluate,args,count,agent_n,epsilon,address,queue=None):
    
    agent_n_cpu = QMIX_AIGC_CPU(args)
    agent_n_cpu.eval_Q_net_cpu.load_state_dict(agent_n.eval_Q_net_cpu.state_dict())
    
    env = AIGC2Env(address=address)
    
    replay_buffers = []
    win_tags = []
    episode_rewards = []
    episode_steps = []
    each_work_episode = args.each_work_episode
    if evaluate:
        each_work_episode = 1
    for _ in range(each_work_episode):
        if args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            agent_n_cpu.eval_Q_net_cpu.rnn_hidden = None
        if args.use_reward_norm:
            print("------use reward norm------")
            reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:
            print("------use reward scaling------")
            reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        last_onehot_a_n = np.zeros((args.N, args.action_dim))  # Last actions of N agents(one-hot)
        replay_buffer = []
        env._reset()
        win_tag = False
        episode_reward = 0
        for episode_step in range(args.episode_limit+env.start_time):
            if ISHOST:
                env.print_logs(count)
            obs_n = env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = env.get_state()  # s.shape=(state_dim,)
            
            avail_a_n = env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else epsilon
            a_n = agent_n_cpu.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            
            last_onehot_a_n = np.eye(args.action_dim)[a_n]  # Convert actions to one-hot vectors
            r, done, info, fixed_act = env._step(a_n)  # Take a step
            if episode_step < env.start_time:
                continue
            r = sum(r)/10
            win_tag = True if all(done) and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if all(done) and episode_step + 1 != args.episode_limit:
                    dw = True
                else:
                    dw = False
                # Store the transition
                replay_buffer.append((episode_step-env.start_time, obs_n, s, avail_a_n, last_onehot_a_n, a_n, fixed_act, r, done, dw))
                # Decay the epsilon
                epsilon = epsilon - args.epsilon_decay if epsilon - args.epsilon_decay > args.epsilon_min else args.epsilon_min
            if all(done):
                break
        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = env.get_obs()
            s = env.get_state()
            avail_a_n = env.get_avail_actions()
            replay_buffer.append((episode_step + 1 - env.start_time, obs_n, s, avail_a_n))
            replay_buffers.append(replay_buffer)
            win_tags.append(win_tag)
            episode_steps.append(episode_step + 1 - env.start_time)
            episode_rewards.append(episode_reward)
    env.close()
    if not evaluate:
        if queue is not None:
            queue.put((replay_buffers,win_tags, episode_rewards, episode_steps, epsilon))
        return replay_buffers,win_tags, episode_rewards, episode_steps, epsilon
    else:
        if queue is not None:
            queue.put((win_tags, episode_rewards, episode_steps))
        return win_tags, episode_rewards, episode_steps

class Runner_QMIX_AIGC:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.best_eva_rewards = 0
        self.count = 0
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.address = ADDRESS['ip'] + ":" + str(ADDRESS['port'])
        env = AIGC2Env(address=self.address)
        self.env_info = env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]+10  # Maximum number of steps per episode
        env.end()
        env.close()
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # multiprocessing
        self.num_workers = args.num_workers

        self.args.each_work_episode = int(self.args.batch_size / self.num_workers)
        # 设置启动方法为'spawn'
        mp.set_start_method('spawn')
        # Create N agents
        self.agent_n = QMIX_AIGC(self.args)
        self.agent_n_cpu = QMIX_AIGC_CPU(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='agent/QMIX_VDN_AIGC/runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        # load checkpoint
        # self.total_steps, self.best_eva_rewards = self.agent_n.load_model(env_name, self.args.algorithm, number, 1026, 251)    
        
    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
            # 创建进程池和队列
            pool = mp.Pool(processes=self.num_workers)
            manager = mp.Manager()
            queue = manager.Queue() 
            # 采样
            evaluate = False
            self.agent_n_cpu.eval_Q_net_cpu.load_state_dict(self.agent_n.eval_Q_net.state_dict())
            processes = [pool.apply_async(run_episode_aigc, args=(evaluate,self.args,self.count+i,self.agent_n_cpu,self.epsilon,ADDRESS['ip'] + ":" + str(int(ADDRESS['port']) + i),queue)) for i in range(self.num_workers)]
            for result in processes:
                replay_buffers, win_tags, episode_rewards, episode_steps, epsilon = result.get()
                self.epsilon = epsilon
                self.replay_buffer.add_episode_replay(replay_buffers)
                self.total_steps += np.array(episode_steps).sum()
                if self.replay_buffer.episode_num == self.args.batch_size:
                    a_l = self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                    self.writer.add_scalar('agent_loss_{}'.format(self.env_name), a_l, global_step=self.total_steps)
            # 关闭进程池
            pool.close()
            pool.join()  
        self.evaluate_policy()
 
    def evaluate_policy(self, train=True):
        win_times = 0
        evaluate_reward = 0
        # 创建进程池和队列
        pool = mp.Pool(processes=self.args.evaluate_times)
        manager = mp.Manager()
        queue = manager.Queue() 
        # 采样
        evaluate = True
        self.agent_n_cpu.eval_Q_net_cpu.load_state_dict(self.agent_n.eval_Q_net.state_dict())
        processes = [pool.apply_async(run_episode_aigc, args=(evaluate,self.args,self.count+i,self.agent_n_cpu,self.epsilon,ADDRESS['ip'] + ":" + str(int(ADDRESS['port']) + i),queue)) for i in range(self.args.evaluate_times)]
        self.count += self.args.evaluate_times
        for result in processes:
            win_tags, episode_rewards, episode_steps = result.get()
            win_times += win_tags.count(True)
            evaluate_reward += np.array(episode_rewards).sum()
        # 关闭进程池
        pool.close()
        pool.join()
        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        if train:
            self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
            self.writer.add_scalar('evaluate_reward_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
            # Save the win rates
            if self.best_eva_rewards<evaluate_reward:
                self.best_eva_rewards=evaluate_reward
                self.agent_n.save_model(self.env_name, self.args.algorithm, self.number, evaluate_reward, self.total_steps)
