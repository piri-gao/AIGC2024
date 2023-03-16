from typing import List
from agent.agent import Agent
from env.env_cmd import CmdEnv
import numpy as np
import math 
import os
import sys
from agent.QYKZ_semi.style_agent_a import QYKZ_Agent as Agent_style_a
from agent.QYKZ_semi.style_agent_b import DemoAgent as Agent_style_b
import re
import shutil
import random

count = 1
Init_clear = False
K = 2
rewards = [0] * K
Epsilon = 0.9
Epsilon_min = 0.1
Epsilon_decay_rate = 0.95
mode = "e-greedy"


class UCB1:
    """待调试"""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)   # 每个臂的选择次数
        self.values = np.zeros(n_arms)   # 每个臂的平均奖励值
        self.total_counts = 0            # 所有臂的选择总次数

    def select_arm(self):
        # 每个臂至少被选择一次
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # 计算UCB值并选择UCB值最大的臂
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_counts) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        # 更新对应臂的选择次数、平均奖励值以及所有臂的选择总次数
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value
        self.total_counts += 1  

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)

    def select_arm(self):
        # 对每个臂采样
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        # 选择采样值最大的臂
        return np.argmax(samples)

    def update(self, arm, reward):
        # 更新对应臂的成功和失败次数
        if reward:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1


class Softmax:
    def __init__(self, n_arms, temperature):
        self.n_arms = n_arms
        self.temperature = temperature
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        # 计算softmax分布
        exp_values = np.exp(self.values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        # 选择概率最大的臂
        return np.random.choice(range(self.n_arms), p=probs)

    def update(self, arm, reward):
        # 更新对应臂的计数和价值
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value


class QYKZ_Agent(Agent):
    """
        自定义智能体
    """
    def __init__(self, name, config):
        self.style_a = Agent_style_a(name, config)
        self.style_b = Agent_style_b(name, config)
        self.id = 0
        self.max_id = 2
        self.alpha = 0.5
        self.epsilon = 0
        self.win_prob = [0.5, 0.5]
        if name == 'red':
            self.name = "red"
        else:
            self.name = "blue"
        #指定要读的文件
        from config import config as config_dict
        self.red_cls = config_dict["agents"]['red']
        self.blue_cls = config_dict["agents"]['blue']
        #self.count = 1 #对局计数
        # print("重新初始化") #每次都会重新初始化
        self.new_done = 0

        # 不同的策略选择算法
        self.ucb1 = UCB1(K)
        self._init()
    
    def _init(self):
        global count 
        global Init_clear
        global rewards
        global Epsilon
        global Epsilon_min
        global Epsilon_decay_rate
        global K
        pre_name = str(self.red_cls).split("'")[1] + "_VS_" + str(self.blue_cls).split("'")[1]
        self.file_name = "logs/" + pre_name + "_" + str(
            count) + ".txt"
        #初始化时 删除之前的txt文件
        if not Init_clear:
            Init_clear = True
            for root, dirs, files in os.walk("logs"):
                for file in files:
                    if file.startswith(pre_name) and file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)

        #读取结果 按照结果选定本轮要用的策略
        if mode == "e-greedy":
            # 已测试
            print(f'本局epsilon:{Epsilon}')
            if Epsilon > Epsilon_min:
                Epsilon *= Epsilon_decay_rate
            if random.uniform(0,1) < Epsilon:
                self.id = random.randint(0, K-1)
                print(f'本局随机选中的策略ID为:{self.id}')
            else:
                max_reward = max(rewards)
                indices = [i for i in range(K) if rewards[i] == max_reward]
                self.id = random.choice(indices)
                print(f'本局选中的最优策略ID为:{self.id}')
        if mode == "ucb1":
            # 待调试
            arm = self.ucb1.select_arm(arm)
            reward = rewards[arm]
            self.ucb1.update(arm, reward)


        #print(f"本轮选定的策略为:{self.id}")

    def reset(self):
        self.style_a.reset()
        self.style_b.reset()
        

    



    def step(self, sim_time, obs, **kwargs) -> List[dict]:
        global count
        global rewards
        """执行相应策略的动作 若结束则将结果进行存储"""
        if not os.path.exists(self.file_name):
            pass
        else:
            with open(self.file_name, "r") as f:
                lines = f.readlines()
                if "蓝方得分" in lines[-1]:
                    self.new_done += 1
                    # 该对局已结束
                    # 匹配数字
                    blue_score = int(re.search(r'\d+', lines[-1]).group(0))
                    red_score = int(re.search(r'\d+', lines[-2]).group(0))
                    if self.new_done == 1:
                        print(f"当前为第{count}局，红方得分{red_score}")
                        print(f"当前为第{count}局，蓝方得分{blue_score}")
                        print(f"本局持{self.name}方")
                        if "双方得分相同" not in lines[-4]:
                            if self.name == "red":
                                rewards[self.id] += int(red_score>blue_score)
                                rewards[self.id] += 0.5 * int(red_score == blue_score)
                            if self.name == "blue":
                                rewards[self.id] += int(blue_score>red_score)
                                rewards[self.id] += 0.5 * int(red_score == blue_score)
                        else:
                            if ("蓝方获胜" in lines[-3]) and (self.name == "blue"):
                                rewards[self.id] += 1
                            if ("红方获胜" in lines[-3]) and (self.name == "red"):
                                rewards[self.id] += 1
                        print(f"更新后各模型奖励:{rewards}")
                        print('--------------------------------------------')
                    # 统计当前文件夹下对战局数
                    episode_num = 0
                    pre_name = str(self.red_cls).split("'")[1] + "_VS_" + str(self.blue_cls).split("'")[1]
                    for root, dirs, files in os.walk("logs"):
                        for file in files:
                            if file.startswith(pre_name) and file.endswith(".txt"):
                                episode_num += 1 
                    #print("count:",count)
                    #print("episode_num:",episode_num)
                    if count == episode_num-1:
                        count += 1
                    
        if self.id == 0:
            return self.style_a.step(sim_time, obs, **kwargs)
        if self.id == 1:
            return self.style_b.step(sim_time, obs, **kwargs)


