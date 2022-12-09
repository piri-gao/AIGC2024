# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-10-21 08:01:33
# @Last Modified by:   Your name
# @Last Modified time: 2021-10-21 08:03:39
import random
from typing import List
from agent.agent import Agent
from  env.env_cmd import CmdEnv
from utils.utils_math import TSVector3

from agent.modify_v1 import Modify as aa
from agent.LiuBY_Phoenix_3 import LiuBY_Phoenix_Agent as bb
from agent.infinite import LiuBY_Infinite_Agent as dd
from agent.rl_agentjyh.rl_agent import RL_Agent as cc

import math
import os
import struct



class mixAgent(Agent):
    def __init__(self, name, config):
        self.a1 = aa(name, config)
        self.a2 = bb(name, config)
        self.a3 = cc(name, config)
        self.a4 = dd(name, config)
        # self.a5 = ee(name, config)
        self.id = 1
        self.max_id = 4 # 共有几个AI参与混合
        self.alpha = 0.5 # a步长更新
        self.e = 0 # e贪心
        self.win_poss = [1, 0.9, 0.8, 0.7, 0.6]
        if name == 'red':  #0红 1蓝
            self.name = 0
        else:
            self.name = 1
        self.red_missiles = ['空空导弹_1(红有人机_武器系统_1)','空空导弹_2(红有人机_武器系统_1)',\
                             '空空导弹_3(红有人机_武器系统_1)','空空导弹_4(红有人机_武器系统_1)',\
                             '空空导弹_1(红无人机1_武器系统_1)','空空导弹_2(红无人机1_武器系统_1)',\
                             '空空导弹_1(红无人机2_武器系统_1)','空空导弹_2(红无人机2_武器系统_1)',\
                             '空空导弹_1(红无人机3_武器系统_1)','空空导弹_2(红无人机3_武器系统_1)',\
                             '空空导弹_1(红无人机4_武器系统_1)','空空导弹_2(红无人机4_武器系统_1)']
        self.blue_missiles = ['空空导弹_1(蓝有人机_武器系统_1)','空空导弹_2(蓝有人机_武器系统_1)',\
                              '空空导弹_3(蓝有人机_武器系统_1)','空空导弹_4(蓝有人机_武器系统_1)',\
                              '空空导弹_1(蓝无人机1_武器系统_1)','空空导弹_2(蓝无人机1_武器系统_1)',\
                              '空空导弹_1(蓝无人机2_武器系统_1)','空空导弹_2(蓝无人机2_武器系统_1)',\
                              '空空导弹_1(蓝无人机3_武器系统_1)','空空导弹_2(蓝无人机3_武器系统_1)',\
                              '空空导弹_1(蓝无人机4_武器系统_1)','空空导弹_2(蓝无人机4_武器系统_1)']
        self._init()

    def _init(self):
        self.cur_time = 0
        self.result_file = os.path.dirname(__file__) + '/result.txt'
        with open(self.result_file,'r') as f:
            all_obs = f.readlines()
        self.result = self.get_result(all_obs) #0红胜利 1蓝胜利
        #print(self.id, self.name, self.result)
        if self.name == self.result:
            self.win_poss[self.id - 1] = (1 - self.alpha) * self.win_poss[self.id - 1] + self.alpha
        else:
            self.win_poss[self.id - 1] = (1 - self.alpha) * self.win_poss[self.id - 1]
        for i in range(self.max_id):
            if self.win_poss[i] > self.win_poss[self.id - 1]:
                self.id = i + 1
        #m = random.randint(0, 10000)
        #if m < 10000 * self.e:
        #    self.id = random.randint(1, 4)
        #print(self.id, self.win_poss)


    def reset(self, **kwargs):
        self.a1.reset()
        self.a2.reset()
        self.a3.reset()
        self.a4.reset()
        # self.a5.reset()
        self.cur_time = 0
        with open(self.result_file,'r') as f:
            all_obs = f.readlines()
        self.result = self.get_result(all_obs)  # 0红胜利 1蓝胜利
        with open(self.result_file,'w') as f:
            f.write('')
        #print(self.id, self.name, self.result)
        if self.name == self.result:
            self.win_poss[self.id - 1] = (1 - self.alpha) * self.win_poss[self.id - 1] + self.alpha
        else:
            self.win_poss[self.id - 1] = (1 - self.alpha) * self.win_poss[self.id - 1]
        for i in range(self.max_id):
            if self.win_poss[i] > self.win_poss[self.id - 1]:
                self.id = i + 1
        #m = random.randint(0, 10000)
        #if m < 10000 * self.e:
        #    self.id = random.randint(1, 4)
        #print(self.id, self.win_poss)
        self.red_missiles = ['空空导弹_1(红有人机_武器系统_1)','空空导弹_2(红有人机_武器系统_1)',\
                             '空空导弹_3(红有人机_武器系统_1)','空空导弹_4(红有人机_武器系统_1)',\
                             '空空导弹_1(红无人机1_武器系统_1)','空空导弹_2(红无人机1_武器系统_1)',\
                             '空空导弹_1(红无人机2_武器系统_1)','空空导弹_2(红无人机2_武器系统_1)',\
                             '空空导弹_1(红无人机3_武器系统_1)','空空导弹_2(红无人机3_武器系统_1)',\
                             '空空导弹_1(红无人机4_武器系统_1)','空空导弹_2(红无人机4_武器系统_1)']
        self.blue_missiles = ['空空导弹_1(蓝有人机_武器系统_1)','空空导弹_2(蓝有人机_武器系统_1)',\
                              '空空导弹_3(蓝有人机_武器系统_1)','空空导弹_4(蓝有人机_武器系统_1)',\
                              '空空导弹_1(蓝无人机1_武器系统_1)','空空导弹_2(蓝无人机1_武器系统_1)',\
                              '空空导弹_1(蓝无人机2_武器系统_1)','空空导弹_2(蓝无人机2_武器系统_1)',\
                              '空空导弹_1(蓝无人机3_武器系统_1)','空空导弹_2(蓝无人机3_武器系统_1)',\
                              '空空导弹_1(蓝无人机4_武器系统_1)','空空导弹_2(蓝无人机4_武器系统_1)']
       

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        self.cur_time += 1
        with open(self.result_file,'a') as f:
            f.writelines(str(obs_side))

        if self.id == 1:
            return self.a1.step(sim_time, obs_side, **kwargs)
        elif self.id == 2:
            return self.a2.step(sim_time, obs_side, **kwargs)
        elif self.id == 3:
            return self.a3.step(sim_time, obs_side, **kwargs)
        elif self.id == 4:
            return self.a4.step(sim_time, obs_side, **kwargs)
        # elif self.id == 5:
        #     return self.a5.step(sim_time, obs_side, **kwargs)

    def get_result(self, obs):
        if len(obs) == 0:
            return self.name
        else:
            game_info = obs[0].split('{\'platforminfos\':')
            platform_info = []
            track_info = []
            missile_info = []
            red_youren_distance = []
            blue_youren_distance = []
            for item in game_info:
                if len(item) > 1:
                    step_info = item.split('[')
                    platform_info.append(step_info[1])
                    track_info.append(step_info[2])
                    missile_info.append(step_info[3])

            if len(game_info) <= 1198:
                #判断是否战损
                last_platform_info = platform_info[-1]
                last_platform_youren = last_platform_info.split('Name')[1]
                last_platform_youren_info = last_platform_youren.split(',')
                x0 = last_platform_youren_info[5][5:]
                y0 = last_platform_youren_info[6][5:]
                z0 = last_platform_youren_info[9][7:]

                last_track_info = track_info[-1]
                last_track_youren = last_track_info.split('Name')[1]
                last_track_youren_info = last_track_youren.split(',')
                x1 = last_track_youren_info[5][5:]
                y1 = last_track_youren_info[6][5:]
                z1 = last_track_youren_info[9][7:]

                last_missile_info = missile_info[-1]
                last_step_missiles = last_missile_info.split('Name')[1:]
                for last_step_missile in last_step_missiles:
                    last_step_missile_info = last_step_missile.split(',')
                    x = last_step_missile_info[5][5:]
                    y = last_step_missile_info[6][5:]
                    z = last_step_missile_info[9][7:]

                    x = float(x)
                    y = float(y)
                    z = float(z)
                    x0 = float(x0)
                    y0 = float(y0)
                    z0 = float(z0)
                    x1 = float(x1)
                    y1 = float(y1)
                    z1 = float(z1)

                    if self.name == 0:
                        if '蓝' in last_step_missile_info[0]: #x0是红方有人机,x是蓝方导弹
                            distance = math.sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))
                            red_youren_distance.append(distance)
                        if '红' in last_step_missile_info[0]:#x1是蓝方有人机，x是红方导弹
                            distance = math.sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1))
                            blue_youren_distance.append(distance)
                    if self.name == 1:
                        if '蓝' in last_step_missile_info[0]: #x1是红方有人机
                            distance = math.sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1))
                            red_youren_distance.append(distance)
                        if '红' in last_step_missile_info[0]:#x0是蓝方有人机
                            distance = math.sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))
                            blue_youren_distance.append(distance)
                if len(blue_youren_distance) > 0 and len(red_youren_distance) > 0:
                    if min(blue_youren_distance) < min(red_youren_distance):
                       
                        return 0
                    if min(blue_youren_distance) >= min(red_youren_distance):
                       
                        return 1
                if len(blue_youren_distance) == 0:
                   
                    return 1
                if len(red_youren_distance) == 0:
                 
                    return 0

                #判断是否无弹
                '''
                for step_missile_info in missile_info:
                    for red_missile in self.red_missiles:
                        if red_missile in step_missile_info:
                            self.red_missiles.remove(red_missile)
                    for blue_missile in self.blue_missiles:
                        if blue_missile in step_missile_info:
                            self.blue_missiles.remove(blue_missile)
                '''
                red_fly_missile_num = 0
                blue_fly_missile_num = 0
                for step_missile_info in missile_info:
                    if step_missile_info['Identification'] == '红方':
                        red_fly_missile_num += 1
                    if step_missile_info['Identification'] == '蓝方':
                        blue_fly_missile_num += 1
                    for red_missile in self.red_missiles[::-1]:
                        if red_missile in step_missile_info:
                            self.red_missiles.remove(red_missile)
                    for blue_missile in self.blue_missiles[::-1]:
                        if blue_missile in step_missile_info:
                            self.blue_missiles.remove(blue_missile)
                '''
                if len(self.red_missiles) == 0:
                    print('红无弹')
                    return 1
                if len(self.blue_missiles) == 0:
                    print('蓝无弹')
                    return 0
                '''
                if len(self.red_missiles) == 0 and red_fly_missile_num == 0:
                  
                    return 1
                if len(self.blue_missiles) == 0 and blue_fly_missile_num == 0:
                 
                    return 0


            else:
                # 计算导弹积分、飞机积分
                #导弹积分
                for step_missile_info in missile_info:
                    for red_missile in self.red_missiles[::-1]:
                        if red_missile in step_missile_info:
                            self.red_missiles.remove(red_missile)
                    for blue_missile in self.blue_missiles[::-1]:
                        if blue_missile in step_missile_info:
                            self.blue_missiles.remove(blue_missile)
                red_missile_score = (12-len(self.red_missiles))*(-3)
                blue_missile_score = (12-len(self.blue_missiles))*(-3)

                #飞机积分
                last_platform_info = platform_info[-1]
                a = last_platform_info.count('无人机')
                platform_plane_score = (4 - last_platform_info.count('无人机'))*(-15)
                last_track_info = track_info[-1]
                track_plane_score = (4 - last_track_info.count('无人机')) * (-15)
                if self.name == 0:
                    red_plane_score = platform_plane_score
                    blue_plane_score = track_plane_score
                if self.name == 1:
                    red_plane_score = track_plane_score
                    blue_plane_score = platform_plane_score

                red_score = red_missile_score + red_plane_score
                blue_score = blue_missile_score + blue_plane_score
                if red_score > blue_score:
                  
                    return 0
                if red_score < blue_score:
               
                    return 1

                #计算制空积分
                platform_score = 0
                for step_platform_info in platform_info:
                    step_platform_youren  = step_platform_info.split('Name')[1]
                    step_platform_youren_info = step_platform_youren.split(',')
                    x0 = step_platform_youren_info[5][5:]
                    y0 = step_platform_youren_info[6][5:]
                    z0 = step_platform_youren_info[9][7:]
                    x0 = float(x0)
                    y0 = float(y0)
                    z0 = float(z0)
                    distance_to_center = math.sqrt(x0*x0 + y0*y0 + (z0-9000)*(z0-9000))
                    if distance_to_center <= 50000 and z0 >= 2000 and z0 <= 16000:
                        platform_score = platform_score + 1

                track_score = 0
                for step_track_info in track_info:
                    step_track_youren  = step_track_info.split('Name')[1]
                    step_track_youren_info = step_track_youren.split(',')
                    x1 = step_track_youren_info[5][5:]
                    y1 = step_track_youren_info[6][5:]
                    z1 = step_track_youren_info[9][7:]
                    x1 = float(x1)
                    y1 = float(y1)
                    z1 = float(z1)
                    distance_to_center = math.sqrt(x1*x1 + y1*y1 + (z1-9000)*(z1-9000))
                    if distance_to_center <= 50000 and z1 >= 2000 and z1 <= 16000:
                        track_score = track_score + 1

                if self.name == 0:
                    red_score = platform_score
                    blue_score = track_score
                if self.name == 1:
                    red_score = track_score
                    blue_score = platform_score

                if red_score >= blue_score:
               
                    return 0
                if red_score < blue_score:
                
                    return 1


