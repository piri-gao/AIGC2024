# -*- coding: utf-8 -*-
# @Author: JYH & LBY
# @Date:   2021-10-03 23:18:05
# @Last Modified by:   Your name
# @Last Modified time: 2021-10-20 23:55:58

from typing import List
from agent.agent import Agent
import pickle

from .model import rl_model as rl_model
from .ibll.aig_feature import Feature
from .ibll.aig_feature2 import Feature as Feature2
import numpy as np
from .ibll.process import processor

MODEL_PATH = "agent/rl_agentjyh/model/highest_23_0.62.pth"

class RL_Agent(Agent):
    def __init__(self, name, config):
        super(RL_Agent, self).__init__(name, config["side"])
        self.Model = rl_model.create_rlmodel(1)
        if name == "red":
            self.Feature = Feature()
        else:
            self.Feature = Feature2()
        self.my_color = config["side"]
        self.Process = processor(self.my_color)
        file_path = MODEL_PATH
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
        self.set_weights(weights)

    def set_weights(self, weights):
        self.Model.set_weights(weights)
    
    def get_weights(self):
        w = self.Model.get_weights()
        return w

    def reset(self, **kwargs):
        # TODO 各个模块是否需要reset
        self.Feature = Feature()
        self.Process = processor(self.my_color)

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        cmd_list = []
        self.process_decision(sim_time, obs_side, cmd_list)
        return cmd_list
    
    def process_decision(self, sim_time, obs_side, cmd_list):
        cmds = []
        if sim_time == 1:  # 当作战时间为1s时,初始化实体位置,注意,初始化位置的指令只能在前三秒内才会被执行
            cmds = self.Process.Action2cmd(obs_side, sim_time, None) # TODO 输入None时执行init动作
        elif sim_time >= 2:  # 当作战时间大于10s时,开始进行任务控制,并保存任务指令;
            feature = self.Feature.process(sim_time, obs_side) 
            feature_expand = np.expand_dims(feature, 0)

            mask = self.Process.Mask(sim_time, obs_side)

            _, pi = self.Model.predict(feature_expand, batch_size=1) # 1, n, a
            pi = np.squeeze(pi, axis = 0)  # n, a

            action = self.Process.choose_action(pi, mask, train=False)
            # print("action", action)
            cmds = self.Process.Action2cmd(obs_side, sim_time, action)

        cmd_list += cmds
    
    def step_train(self, sim_time, obs_side, feature):
        cmd_list = []
        pi, one_hot_action, mask, value = self.process_decision_train(sim_time, obs_side, cmd_list, feature)
        return cmd_list, pi, one_hot_action, mask, value
    
    def process_decision_train(self, sim_time, obs_side, cmd_list, feature):
        cmds = []
        pi = np.zeros([5, 6])
        one_hot_action = np.zeros([5, 6])
        mask = np.zeros([5, 6])
        value = np.zeros([5])
        if sim_time == 1:
            cmds = self.Process.Action2cmd(obs_side, sim_time, actions=None) # 输入action为None时，输出cmd为init_pos相关指令
        elif sim_time >= 2:
            feature_expand = np.expand_dims(feature, 0)
            mask = self.Process.Mask(sim_time, obs_side)

            value, pi = self.Model.predict(feature_expand, batch_size=1) # 1, n, a
            pi = np.squeeze(pi, axis = 0)  # n, a TODO
            value = np.squeeze(value, axis = 0)  # n TODO
            
            actions = self.Process.choose_action(pi, mask, train=True)
            dead = np.argwhere(actions==-1)
            if len(dead):
                actions[dead] = 0
                one_hot_action = np.eye(6, 6)[actions]
                one_hot_action = np.array(one_hot_action)
                one_hot_action[dead] = np.zeros(6)
            else:
                one_hot_action = np.eye(6, 6)[actions]  # n, 6
            pi = np.array(pi)
            value = np.array(value)
            one_hot_action = np.array(one_hot_action)
            cmds = self.Process.Action2cmd(obs_side, sim_time, actions)
        
        cmd_list += cmds
        return pi, one_hot_action, mask, value