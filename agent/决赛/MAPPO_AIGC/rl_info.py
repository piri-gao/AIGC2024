import math
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.agent import Agent
import numpy as np
import random
from agent.MAPPO_AIGC.agent_base import Plane, Missile
class DemoAgent(Agent):

    def __init__(self, name, config):
        super(DemoAgent, self).__init__(name, config['side'])
        self._init()

    def _init(self):
        # 我方所有飞机信息列表
        self.my_plane = []
        # 我方有人机信息列表
        self.my_leader_plane = []
        # 我方无人机信息列表
        self.my_uav_plane =[]
        # 是否复仇
        self.revenge = False
        # 攻击距离
        self.attack_distance = 15000
        # 敌方所有飞机信息列表
        self.enemy_plane = []
        # 敌方有人机信息列表
        self.enemy_leader_plane = []
        # 敌方无人机信息列表
        self.enemy_uav_plane = []
        # 导弹信息
        self.missile_list = []
        # 识别红军还是蓝军
        self.side = None
        # 第一编队
        self.first_formation = []
        # 当前时间
        self.sim_time = 0
        # 当前可发射导弹
        self.ready_weapon = 0
        # 当前敌有人机还剩多少
        self.live_enemy_leader = 0
        self.missile_ratio = None
        # 判定赢的分数
        self.win_score = 16-0.2*24
        # 敌方分数是否变化
        self.enemy_score_before = 0
        # 我方分数
        self.my_score = 0
        # 敌方分数
        self.enemy_score = 0
        # 导弹分数
        self.missile_score = 0.2
        # 有人机分数
        self.leader_score = 16
        # 我方已爆炸导弹
        self.finish_missile = []
        # 无人机分数
        self.uav_score = 4
        # 每个智能体的奖励
        self.rewards = [0 for i in range(10)]
        # 智能体done信息
        self.done = [0 for i in range(10)]
        # 我方空闲飞机
        self.free_plane = []
        # 下标与ID转换
        self.index_to_id = {}
        # RL空间heading 和 pitch转换矩阵
        self.heading_pitch = None
        # heading划分
        self.num_azimuth=4
        # pitch划分
        self.num_elevation=3
        self.EPS = 0.004

    def reset(self):
        self._init()

    def step(self,sim_time, obs_side, **kwargs):
        self.sim_time = sim_time
        my_obs,global_obs,reward,done,actions_mask,info=self.update_rl_info(obs_side)
        return my_obs,global_obs,reward,done,actions_mask,info

    # 更新当前各个实体的存亡以及部分额外记录信息
    def update_dead_info(self):
        self.rewards = [0 for i in range(10)]
        for i, plane in enumerate(self.my_plane):
            if plane.i == -1:
                plane.i = i
                self.index_to_id[i] = plane.ID
            plane.move_order = None
            plane.be_seen_leader = False
            plane.be_seen = False
            plane.be_seen_myleader = False
            plane.ready_attack = []
            plane.be_in_danger = False
            if self.sim_time<20:
                plane.jammed = -100
            if plane.follow_plane:
                enemy = self.get_body_info_by_id(self.enemy_plane, plane.follow_plane)
                if plane.ID not in enemy.followed_plane or plane.ready_missile==0:
                    plane.follow_plane = None
            if plane.Type==1 and self.sim_time-plane.last_jam>0:
                plane.do_jam = False
                plane.do_turn = False
            if plane.lost_flag and plane.Availability:
                plane.Availability = 0
                self.done[i] = 1
                self.rewards[i] -= 2 if plane.Type==2 else 4
                if plane.follow_plane is not None:
                    enemy = self.get_body_info_by_id(self.enemy_plane, plane.follow_plane)
                    if plane.ID in enemy.followed_plane:
                        enemy.followed_plane.remove(plane.ID)
                    plane.follow_plane = None
            elif plane.Availability and self.is_in_center(plane):
                plane.center_time += 1
                self.rewards[i] += 0.1
            elif plane.Availability:
                self.rewards[i] += 0.2
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                    self.rewards[i] += 1
                    if plane.do_be_fired:
                        missile.be_fired = True
                if missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list and missile.lost_flag==0 and missile.loss_target==False:
                    plane.locked_missile_list.append(missile.ID)
                    self.rewards[i] -= 0.02
                if missile.EngageTargetID == plane.ID and missile.loss_target==False:
                    short_dis, missile.arrive_time = self.shortest_distance_between_linesegment(plane, missile)
            plane.do_be_fired = False
            # 更新导弹追踪信息
            for missile_id in plane.locked_missile_list.copy():
                tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                be_seen = False
                for m_p in self.my_plane:
                    if m_p.can_see(tmp_missile, see_factor=0.97) and m_p.lost_flag==0:
                        be_seen = True
                if (tmp_missile.lost_flag and (self.sim_time - tmp_missile.arrive_time > 2 or (be_seen and self.sim_time - tmp_missile.arrive_time>1))) or tmp_missile.loss_target:
                    plane.locked_missile_list.remove(tmp_missile.ID)
                    self.rewards[i] += 1
            # 找到最近的导弹
            dis = 99999999
            plane.close_missile = None
            for missile_id in plane.locked_missile_list:
                missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if TSVector3.distance(plane.pos3d, missile.pos3d) < dis and missile.loss_target==False:
                    dis = TSVector3.distance(plane.pos3d, missile.pos3d)
                    plane.close_missile = missile  
        for plane in self.my_plane:
            if plane.Availability:
                for other_plane in self.my_plane:
                    if other_plane.ID != plane.ID and other_plane.Availability and other_plane.can_see(plane, see_factor=0.99):
                        plane.be_seen = True
        # 更新是否反击
        self.win_now()
        change_dis = False
        self.live_enemy_leader = 0
        for plane in self.enemy_plane:
            plane.ready_attacked = []
            plane.be_seen_leader = False
            plane.be_seen_myleader = False
            plane.be_seen = False
            if self.sim_time<20:
                plane.jammed = -100
            if plane.lost_flag:
                if len(plane.locked_missile_list)>0:
                    dead_flag = 0
                    closer_missile = None
                    dis = 999999999
                    for missile_id in plane.locked_missile_list:
                        tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                        if TSVector3.distance(tmp_missile.pos3d,plane.pos3d)<dis and tmp_missile.lost_flag==0 and tmp_missile.loss_target==False:
                            dis = TSVector3.distance(tmp_missile.pos3d, plane.pos3d)
                            closer_missile = tmp_missile
                        if tmp_missile.lost_flag==0 and tmp_missile.loss_target==True:
                            dead_flag = 1
                        if tmp_missile.lost_flag and tmp_missile.ID not in self.finish_missile:
                            for m_p in self.my_plane:
                                if m_p.can_see(plane, see_factor=0.9) and plane.lost_flag==1 and self.death_to_death(plane,tmp_missile,radius=120):
                                    dead_flag = 1
                            self.finish_missile.append(tmp_missile.ID)
                            if tmp_missile.init_dis<self.attack_distance+1000:
                                change_dis = True
                        if dead_flag:
                            attack_plane = self.get_body_info_by_id(self.my_plane, tmp_missile.LauncherID)
                            self.rewards[attack_plane.i] += 1.5 if plane.Type==2 else 3
                    if closer_missile != None: # 逆推演敌机的轨迹
                        plane.imaginative_update_agent_info(closer_missile)
                    if dead_flag:
                        plane.Availability = 0
                elif len(plane.lost_missile_list)>0:
                    dead_flag = 0
                    for missile_id in plane.lost_missile_list:
                        tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                        new_dir = TSVector3.calorientation(plane.Heading, plane.Pitch)
                        dis = TSVector3.distance(tmp_missile.pos3d,plane.pos3d)
                        next_plane_pos = TSVector3.plus(plane.pos3d,TSVector3.multscalar(new_dir,dis/tmp_missile.Speed*plane.Speed))
                        # if tmp_missile.lost_flag and self.death_to_death(plane,tmp_missile,next_plane_pos,radius=100):
                        #     dead_flag += 1
                        #     attack_plane = self.get_body_info_by_id(self.my_plane, tmp_missile.LauncherID)
                        #     self.rewards[attack_plane.i] += 2 if plane.Type==2 else 4
                        if tmp_missile.loss_target==True:
                            dead_flag = -1
                            attack_plane = self.get_body_info_by_id(self.my_plane, tmp_missile.LauncherID)
                            self.rewards[attack_plane.i] += 1.5 if plane.Type==2 else 3
                    if dead_flag==-1:
                        plane.Availability = 0
                if self.is_in_legal_area(plane)==False:
                    plane.Availability = 0
                    for i in range(len(self.rewards)):
                        self.rewards[i] += 2 if plane.Type==2 else 4
            else:
                plane.Availability = 1
            if plane.Availability and self.is_in_center(plane):
                plane.center_time += 1
            if plane.Availability and plane.Type==1:
                self.live_enemy_leader += 1
                for other_plane in self.my_plane:
                    if plane.can_see(other_plane) and other_plane.Availability:
                        other_plane.be_seen_leader = True
            # 预测五秒飞机消失后的行踪
            if plane.Availability and plane.lost_flag<6:
                total_dir = TSVector3.calorientation(plane.Heading, 0)
                distance = plane.Speed+plane.plane_acc*0.5
                next_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(total_dir, distance))
                plane.pos3d = next_pos
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                if missile.loss_target==False and missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list and missile.lost_flag==0 and missile.loss_target==False:
                    plane.locked_missile_list.append(missile.ID)
                if missile.EngageTargetID == plane.ID and missile.loss_target==False and missile.lost_flag==0:
                    short_dis, missile.arrive_time = self.shortest_distance_between_linesegment(plane, missile)
            # 更新导弹追踪信息
            for missile_id in plane.locked_missile_list.copy():
                tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if tmp_missile.lost_flag or tmp_missile.loss_target:
                    plane.locked_missile_list.remove(tmp_missile.ID)
                    if tmp_missile.ID not in plane.lost_missile_list:
                        plane.lost_missile_list.append(tmp_missile.ID)
            # 找到最近的导弹
            dis = 99999999
            plane.close_missile = None
            for missile_id in plane.locked_missile_list:
                missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if TSVector3.distance(plane.pos3d, missile.pos3d) < dis and missile.loss_target==False:
                    dis = TSVector3.distance(plane.pos3d, missile.pos3d)
                    plane.close_missile = missile 
            plane.ready_missile = plane.AllWeapon - len(plane.used_missile_list)
        # 计算对方是否存在僚机掩护
        for plane in self.enemy_plane:
            plane.wing_plane = 0
            if plane.Availability:
                for wing_plane in self.enemy_leader_plane:
                    if wing_plane.Availability and plane.ID != wing_plane.ID and wing_plane.can_see(plane,see_factor=1):
                        plane.wing_plane += 1
        
        # 重新规划发射距离
        if change_dis:
            if  self.my_score<self.enemy_score+self.win_score:
                if self.enemy_score_before-self.enemy_score<4 and change_dis:
                    self.attack_distance = (self.attack_distance - 5000)/2 + 5000
                elif self.enemy_score_before-self.enemy_score>=4 and change_dis:
                    self.attack_distance = (self.attack_distance-500)/2+self.attack_distance
                    if self.attack_distance>21000:
                        self.attack_distance = 21000
            else:
                self.attack_distance = 21000
        # 计算威胁系数决定移动速度
        for plane in self.my_plane:
            if plane.Availability:
                if plane.Type==1:
                    plane.move_speed = 250
                else:
                    plane.move_speed = (plane.para["move_min_speed"]+plane.para["move_max_speed"])/2
                plane.threat_ratio = 0
                if len(plane.locked_missile_list):
                    plane.move_speed = plane.para["move_max_speed"]
                    plane.threat_ratio = 1000
                else:
                    for enemy in self.enemy_plane:
                        if enemy.Availability and enemy.can_see(plane,see_factor=1.0) and enemy.ready_missile>0:
                        # if enemy.Availability:
                            enemy_dis = TSVector3.distance(plane.pos3d, enemy.pos3d)
                            speed_factor = math.tanh(math.pow(plane.para['safe_range']/(enemy_dis+2000),2))
                            if enemy_dis<26000:
                                plane.threat_ratio = max(plane.threat_ratio, math.pow(plane.para['safe_range']/(enemy_dis+2000),2))
                            tmp_move_speed = speed_factor*(plane.para["move_max_speed"]-plane.para["move_min_speed"]) + plane.para["move_min_speed"]
                            plane.move_speed = min(max(plane.move_speed, tmp_move_speed),plane.para["move_max_speed"])
            if plane.Type == 1:
                for enemy in self.enemy_plane:
                    if enemy.lost_flag==0 and plane.can_see(enemy,see_factor=0.99,jam_dis=110000):
                        enemy.be_seen_leader = True
                for myplane in self.my_plane:
                    if plane.ID != myplane.ID and myplane.can_see(plane, see_factor=0.99,jam_dis=119000):
                        plane.be_seen_myleader = True
            # 我方是否可以看见我方
            for myplane in self.my_plane:
                if plane.ID != myplane.ID and myplane.Availability and plane.can_see(myplane,see_factor=0.99,jam_dis=plane.para['radar_range']):
                    myplane.be_seen = True  
        # 敌方是否可以看见敌方
        for enemy in self.enemy_plane:
            for other_enemy in self.enemy_plane:
                if other_enemy.ID != enemy.ID and enemy.Availability and other_enemy.can_see(enemy,see_factor=0.99,jam_dis=other_enemy.para['radar_range']):
                    enemy.be_seen = True
                if other_enemy.ID != enemy.ID and enemy.Type==1 and enemy.Availability and enemy.can_see(other_enemy,see_factor=0.99,jam_dis=119000):
                    other_enemy.be_seen_myleader = True
            if enemy.Type==1 and enemy.Availability:
                for myplane in self.my_plane:
                    if  enemy.can_see(myplane, see_factor=0.99,jam_dis=119000):
                        myplane.be_seen_leader = True
            
                
        # 己方有人机信息
        self.my_leader_plane = self.get_body_info_by_type(self.my_plane, 1)
        # 己方无人机信息
        self.my_uav_plane = self.get_body_info_by_type(self.my_plane, 2)
        # 敌方有人机信息
        self.enemy_leader_plane = self.get_body_info_by_type(self.enemy_plane, 1)
        # 敌方无人机信息
        self.enemy_uav_plane = self.get_body_info_by_type(self.enemy_plane, 2)

        if self.sim_time>19*60+57 and self.my_score>self.enemy_score:
            # print("预测赢了:",self.my_score,'   ',self.enemy_score)
            for i in range(len(self.rewards)):
                self.rewards[i] += 4
        elif self.sim_time>19*60+57 and self.my_score<self.enemy_score:
            for i in range(len(self.rewards)):
                self.rewards[i] -= 0.7
        if self.sim_time>19*60+59:
            for i in range(len(self.done)):
                self.done[i] = 1
        if self.revenge == False:  
            self.enemy_left_weapon = 24
            have_enemy = 0
            self.my_left_weapon = 16
            self.my_ready_weapon = 24
            for plane in self.my_plane:
                if plane.Type==2:
                    if plane.Availability:
                        self.my_left_weapon -= len(plane.used_missile_list)
                    else:
                        self.my_left_weapon -= plane.AllWeapon
                if plane.Availability:
                    self.my_ready_weapon -= len(plane.used_missile_list)
                else:
                    self.my_ready_weapon -= plane.AllWeapon
            for plane in self.enemy_plane:
                if plane.Availability:
                    have_enemy = 1
                    self.enemy_left_weapon -= len(plane.used_missile_list)
                else:
                    self.enemy_left_weapon -= plane.AllWeapon
            if have_enemy and (self.enemy_left_weapon == 1 or (self.sim_time > 20 * 60 -300\
                     and self.my_score<self.enemy_score) or self.my_left_weapon == 0):
                self.revenge = True
        self.first_formation = self.my_uav_plane
        if self.revenge:
            self.first_formation = self.my_plane

    # 局部视野更新各实体信息
    def update_entity_info(self, obs_side):
        # 己方所有飞机信息
        self.my_plane = self.get_all_body_list(self.my_plane, obs_side['platforminfos'], Plane, info_type=0)
        # 敌方所有飞机信息
        self.enemy_plane = self.get_all_body_list(self.enemy_plane, obs_side['trackinfos'], Plane)
        # 获取双方导弹信息
        self.missile_list = self.get_all_body_list(self.missile_list, obs_side['missileinfos'], Missile)
        # 更新阵亡飞机
        self.update_dead_info()
        # 获取队伍标识
        if self.side is None:
            if self.my_plane[0].Identification == "红方":
                self.side = 1
            else:
                self.side = -1
            self.init_direction = 90
            if self.side == -1:
                self.init_direction = 270 
        if len(self.enemy_leader_plane)==2:
            self.bound = 40000 
        else:
            self.bound = 140000
    
    # 判断对方是否是策略：保守or激进
    def enemy_strategy(self):
        is_attack = False
        for enemy in self.enemy_plane:
            if len(enemy.used_missile_list)>0:
                is_attack = True
        return is_attack

    # 导弹与飞机匹配
    def plane_to_missile(self, missile, enemy_plane, team_flag):
        dis = 99999999
        target_plane = -1
        missile_pos = {"X": missile['X'], "Y": missile['Y'], "Z": missile['Alt']}
        for plane in enemy_plane:
            missile_dis = TSVector3.distance(missile_pos, plane.pos3d)
            if missile_dis<dis and plane.AllWeapon - len(plane.used_missile_list)>0:
                if team_flag==1:
                    if plane.ready_missile != plane.AllWeapon - len(plane.used_missile_list):
                        dis = missile_dis
                        target_plane = plane.ID
                else:
                    if missile['Speed']<1000:
                        t = (missile['Speed']-398)/98
                        tmp_dis = 0.5*98*t*t+600+t*350
                        if missile_dis<tmp_dis:
                            dis = missile_dis
                            target_plane = plane.ID
                    elif abs(plane.pi_bound(missile['Heading']-TSVector3.calheading(TSVector3.minus(plane.pos3d, missile_pos))))>math.pi*0.5 and missile_dis>5000:
                        dis = missile_dis
                        target_plane = plane.ID
        if target_plane == -1:
            dis = 99999999
            for plane in enemy_plane:
                missile_dis = TSVector3.distance(missile_pos, plane.pos3d)
                if abs(plane.pi_bound(missile['Heading']-TSVector3.calheading(TSVector3.minus(plane.pos3d, missile_pos))))>math.pi*0.3 and missile_dis<dis:
                    dis = missile_dis
                    target_plane = plane.ID
            if target_plane == -1:
                print(team_flag, missile['ID'],' 没找到目标飞机')
        return target_plane
    
    # 更新真实智能体信息
    def get_all_body_list(self, agent_info_list, obs_list, cls, info_type=-1):
        for obs_agent_info in obs_list:
            agent_in_list = False
            for agent_info in agent_info_list:
                if obs_agent_info['ID'] == agent_info.ID:
                    if obs_agent_info['Type'] == 3:
                        if obs_agent_info['Identification'] != self.my_plane[0].Identification:
                            attacked_plane = self.get_body_info_by_id(self.my_plane, obs_agent_info['EngageTargetID'])
                        else:
                            attacked_plane = self.get_body_info_by_id(self.enemy_plane, obs_agent_info['EngageTargetID'])
                        missile_pos3d = {"X": obs_agent_info['X'], "Y": obs_agent_info['Y'], "Z": obs_agent_info['Alt']}
                        obs_agent_info['distance'] = TSVector3.distance(attacked_plane.pos3d, missile_pos3d)
                    agent_info.update_agent_info(obs_agent_info, info_type)
                    agent_in_list = True
                    break
            if not agent_in_list:
                if obs_agent_info['Type'] == 3:
                    if obs_agent_info['Identification'] != self.my_plane[0].Identification:
                        obs_agent_info['LauncherID'] = self.plane_to_missile(obs_agent_info, self.enemy_plane, team_flag=-1)
                        attacked_plane = self.get_body_info_by_id(self.my_plane, obs_agent_info['EngageTargetID'])
                    else:
                        obs_agent_info['LauncherID'] = self.plane_to_missile(obs_agent_info, self.my_plane , team_flag=1)
                        attacked_plane = self.get_body_info_by_id(self.enemy_plane, obs_agent_info['EngageTargetID'])
                    missile_pos3d = {"X": obs_agent_info['X'], "Y": obs_agent_info['Y'], "Z": obs_agent_info['Alt']}
                    obs_agent_info['distance'] = TSVector3.distance(attacked_plane.pos3d, missile_pos3d)
                agent_info_list.append(cls(obs_agent_info))

        # 更改死亡的智能体状态
        for agent_info in agent_info_list:
            agent_in_list = False
            for obs_agent_info in obs_list:
                if obs_agent_info['ID'] == agent_info.ID:
                    agent_in_list = True
                    break
            if not agent_in_list:
                agent_info.lost_flag += 1
                if agent_info.Type==3 and agent_info.Identification!=self.my_plane[0].Identification and agent_info.loss_target==False:
                    attacked_plane = self.get_body_info_by_id(self.my_plane, agent_info.EngageTargetID)
                    if attacked_plane.Availability:
                        target_dis = TSVector3.distance(attacked_plane.pos3d, agent_info.pos3d)
                        if agent_info.Speed<1000:
                            ans_t = attacked_plane.Speed/(agent_info.Speed+agent_info.missile_acc)
                        else:
                            ans_t = attacked_plane.Speed/agent_info.Speed
                        get_point = TSVector3.plus(attacked_plane.pos3d,TSVector3.multscalar(TSVector3.calorientation(attacked_plane.Heading, attacked_plane.Pitch), ans_t*attacked_plane.Speed))
                        target_pitch = TSVector3.calpitch(TSVector3.minus(get_point, agent_info.pos3d))
                        target_heading = TSVector3.calheading(TSVector3.minus(get_point, agent_info.pos3d))
                        if abs(attacked_plane.pi_bound(target_heading-agent_info.Heading))>math.pi/6:
                            agent_info.loss_target = True  
                        else:
                            agent_info.imaginative_update_agent_info(target_dis,target_heading,target_pitch)
                
        return agent_info_list

    # 通过一种类型获取实体信息
    def get_body_info_by_type(self, agent_info_list, agent_type: int):
        agent_list = []
        for agent_info in agent_info_list:
            if agent_info.Type == agent_type:
                agent_list.append(agent_info)
        return agent_list

    # 通过实体ID获取实体信息
    def get_body_info_by_id(self, agent_info_list, agent_id: int) -> object:
        for agent_info in agent_info_list:
            if agent_info.ID == agent_id:
                return agent_info
        return None

    # 通过实体阵营获取实体信息 
    def get_body_info_by_identification(self, agent_info_list, agent_identification):
        my_missile = []
        for rocket in agent_info_list:
            if rocket.Identification == agent_identification:
                my_missile.append(rocket.ID)
        return my_missile
    
    # 更新决策
    def update_decision(self, obs_side):
        cmd_list = []
        self.update_entity_info(obs_side)
        if self.sim_time <= 2:
            self.init_pos(cmd_list)
        else:
            # 更新空闲飞机以及威胁飞机信息模块
            self.free_plane = []
            no_missile_plane = []
            threat_plane_list = []
            for my_plane in self.my_plane:
                if len(my_plane.locked_missile_list)==0 and my_plane.Availability:
                    if my_plane.Type==1 and (my_plane.do_jam or my_plane.do_turn):# 该有人机在执行干扰任务
                        continue
                    self.free_plane.append(my_plane)
                    if my_plane.ready_missile == 0 and my_plane.Type == 2:
                        no_missile_plane.append(my_plane.ID)
            if self.free_plane!=[]:
                self.free_plane = sorted(self.free_plane, key=lambda d: d.ready_missile, reverse=False)
                self.free_plane = [plane.ID for plane in sorted(self.free_plane, key=lambda d: d.Type, reverse=True)]
            for enemy in self.enemy_plane:
                if enemy.lost_flag==0:
                    threat_plane_list.append(enemy.ID)
            if len(threat_plane_list):
                threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).wing_plane, reverse=False)
                threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).ready_missile, reverse=True)
                threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).Type, reverse=False)
                        
            # 为受攻击飞机提供导弹视野支持
            need_seen_plane = []
            for plane in self.my_plane:
                if plane.Availability and len(plane.locked_missile_list):
                    need_seen_plane.append(plane.ID)
                else:
                    if plane.wing_plane!=None and plane.Type==2:
                        wing_plane = self.get_body_info_by_id(self.my_plane, plane.wing_plane)
                        wing_plane.wing_who = None
                        plane.wing_plane = None
            for plane_id in need_seen_plane:
                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                max_arrive_time = 0
                for missile_id in plane.locked_missile_list:
                    missile = self.get_body_info_by_id(self.missile_list, missile_id)
                    arrive_time = missile.arrive_time
                    if arrive_time>max_arrive_time:
                        max_arrive_time = arrive_time
                seen_plane = None
                follow_missile = plane.close_missile
                if plane.wing_plane:
                    wing_plane = self.get_body_info_by_id(self.my_plane, plane.wing_plane)
                    if TSVector3.distance(wing_plane.pos3d, follow_missile.pos3d)>wing_plane.para['radar_range'] or len(wing_plane.locked_missile_list) or wing_plane.ID not in self.free_plane:
                        plane.wing_plane = None
                        wing_plane.wing_who = None
                for see_plane_id in self.free_plane:
                    see_plane = self.get_body_info_by_id(self.my_plane, see_plane_id)
                    if see_plane.can_see(follow_missile, see_factor=0.99) and see_plane.wing_who==None and (seen_plane == None or seen_plane.threat_ratio>see_plane.threat_ratio):
                        seen_plane = see_plane
                if seen_plane:
                    if plane.wing_plane!=None:
                        wing_plane = self.get_body_info_by_id(self.my_plane, plane.wing_plane)
                        if seen_plane.threat_ratio<wing_plane.threat_ratio:
                            plane.wing_plane = seen_plane.ID
                            wing_plane.wing_who = None
                            seen_plane.wing_who = plane.ID
                    else:
                        plane.wing_plane = seen_plane.ID
                        seen_plane.wing_who = plane.ID
                if plane.wing_plane == None:
                    for see_plane_id in self.free_plane:
                        see_plane = self.get_body_info_by_id(self.my_plane, see_plane_id)
                        turn_time = self.get_turn_time(see_plane, follow_missile,see_factor=1)
                        if TSVector3.distance(see_plane.pos3d, follow_missile.pos3d)<see_plane.para['radar_range'] \
                            and turn_time<max_arrive_time and see_plane.wing_who==None \
                            and (seen_plane == None or seen_plane.threat_ratio>see_plane.threat_ratio):
                            seen_plane = see_plane
                    if seen_plane==None:
                        continue
                else:
                    seen_plane = self.get_body_info_by_id(self.my_plane, plane.wing_plane)  
                plane.wing_plane = seen_plane.ID
                seen_plane.wing_who = plane.ID
                if plane.wing_plane in self.free_plane:
                    self.free_plane.remove(plane.wing_plane)
                if follow_missile.lost_flag==0:
                    cmd_list.append(env_cmd.make_followparam(seen_plane.ID, follow_missile.ID, seen_plane.move_speed, seen_plane.para['move_max_acc'], seen_plane.para['move_max_g']))
                else:
                    have_seen_missile = False
                    for missile_id in plane.locked_missile_list:
                        missile =  self.get_body_info_by_id(self.missile_list, missile_id)
                        if missile.lost_flag==0:
                            have_seen_missile = True
                            cmd_list.append(env_cmd.make_followparam(seen_plane.ID, missile.ID, seen_plane.move_speed, seen_plane.para['move_max_acc'], seen_plane.para['move_max_g']))
                            break
                    if have_seen_missile==False:
                        cmd_list.append(env_cmd.make_followparam(seen_plane.ID, plane.ID, seen_plane.move_speed, seen_plane.para['move_max_acc'], seen_plane.para['move_max_g']))
            
            # 开火模块，是否可以在攻击的同时进行干扰
            can_attack_flag = True
            count = 0
            while(can_attack_flag):
                can_attack_flag = False
                can_attack_dict = self.can_attack_plane(threat_plane_list)
                count+=1
                if count>100:
                    print("进入死循环了")
                    break
                for attack_plane_id, threat_ID in can_attack_dict.items():
                    attack_plane = self.get_body_info_by_id(self.my_plane, attack_plane_id)
                    threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_ID)
                    if attack_plane is not None:
                        attack_enemy = 0
                        can_attack_now = True
                        for missile_id in threat_plane.locked_missile_list:
                            missile = self.get_body_info_by_id(self.missile_list, missile_id)
                            if missile.LauncherID == attack_plane.ID:
                                if missile.lost_flag==0:
                                    can_attack_now = False
                                    break
                        for missile_id in threat_plane.lost_missile_list:
                            missile = self.get_body_info_by_id(self.missile_list, missile_id)
                            if missile.LauncherID == attack_plane.ID:
                                attack_enemy = max(missile.arrive_time, attack_enemy)
                        if len(threat_plane.ready_attacked)>1:
                            can_attack_now = False
                        if can_attack_now == False:
                            continue
                        attack_dis = TSVector3.distance(attack_plane.pos3d,threat_plane.pos3d)
                        if self.my_score<self.enemy_score+self.win_score or (self.sim_time>15*60 and \
                                (self.my_score<self.enemy_score or (self.my_score==self.enemy_score \
                                and self.my_center_time<self.enemy_center_time))):# 向死而生，破釜沉舟 
                            cold_time = 0
                            if threat_plane.Type==2:
                                if attack_dis<6000:
                                    cold_time = 1
                        else:
                            if threat_plane.Type==2:
                                if attack_dis<6000:
                                    cold_time = 1
                                else:
                                    cold_time = 2
                            else:
                                cold_time = 1
                        if self.my_score>self.enemy_score+self.win_score:
                            cold_time = 1
                        if self.sim_time - attack_enemy < cold_time:
                            can_attack_now = False
                        if TSVector3.distance(attack_plane.pos3d, threat_plane.pos3d)/1000+self.sim_time+2>=20*60:
                            can_attack_now = False
                        if self.my_score-self.missile_score==self.enemy_score and self.my_center_time<self.enemy_center_time:
                            can_attack_now = False  
                        if self.my_score==self.enemy_score and (self.my_center_time > self.enemy_center_time*2 or self.my_center_time > self.enemy_center_time+(20*60-self.sim_time)*self.live_enemy_leader):
                            can_attack_now = False
                        if can_attack_now:
                            if threat_ID not in attack_plane.ready_attack:
                                attack_plane.ready_attack.append(threat_ID)
                            if attack_plane.ID not in threat_plane.ready_attacked:
                                can_attack_flag = True
                                threat_plane.ready_attacked.append(attack_plane.ID)
                                attack_plane.ready_missile -= 1
                                self.my_score -= self.missile_score
            for threat_plane_id in threat_plane_list:
                threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_plane_id)
                if len(threat_plane.ready_attacked)>0:
                    for attack_plane_id in threat_plane.ready_attacked:
                        factor_fight = 1
                        cmd_list.append(env_cmd.make_attackparam(attack_plane_id, threat_plane_id, factor_fight))
                    threat_plane.ready_attacked = []
            
            # 导弹隐身攻击模块
            jam_for_attack = []
            for leader in self.my_leader_plane:
                # 如果在跑路时也能提供干扰支持
                if leader.Availability and self.sim_time-leader.last_jam>60 and leader.do_jam==False:
                    total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(leader, see_dis=20000, seen=False)
                    if total_dir == {"X": 0, "Y": 0, "Z": 0} or min_dis == 99999999:
                        if len(leader.locked_missile_list)==0:
                            jam_for_attack.append(leader)
                        elif leader.close_missile and TSVector3.distance(leader.close_missile.pos3d, leader.pos3d)>19000:
                            jam_for_attack.append(leader)
            if len(jam_for_attack)>0: 
                my_missile = self.get_body_info_by_identification(self.missile_list, self.my_plane[0].Identification)
                for jam_plane in jam_for_attack:
                    can_jammed_enemy = {}
                    jammed_missile_list = []
                    for missile_id in my_missile.copy():
                        missile = self.get_body_info_by_id(self.missile_list, missile_id)
                        target_plane = self.get_body_info_by_id(self.enemy_plane, missile.EngageTargetID)
                        jam_turn_time = self.get_turn_time(jam_plane, target_plane)
                        can_jam = jam_turn_time*missile.Speed<TSVector3.distance(missile.pos3d, target_plane.pos3d)
                        can_see = TSVector3.distance(jam_plane.pos3d, target_plane.pos3d)<120000-jam_turn_time*target_plane.Speed
                        if target_plane.ID not in can_jammed_enemy and missile.lost_flag==0 and target_plane.lost_flag==0 and can_jam and missile.loss_target==False and can_see:# and self.sim_time-target_plane.jammed>6:
                            can_jammed_enemy[target_plane.ID] = jam_plane.pi_bound(TSVector3.calheading(TSVector3.minus(target_plane.pos3d,jam_plane.pos3d))-jam_plane.Heading)
                            my_missile.remove(missile_id)
                            jammed_missile_list.append(missile_id)
                    can_jammed_enemy = [key for key, value in sorted(can_jammed_enemy.items(), key=lambda d: d[1])]
                    myjam = False
                    if len(can_jammed_enemy)==0:
                        myjam=True
                        can_jammed_enemy = {}
                        for plane in self.my_plane:
                            if plane.ID != jam_plane.ID and jam_plane.can_see(plane,see_factor=0.99,jam_dis=119000) and plane.follow_plane:
                                follow_enemy = self.get_body_info_by_id(self.enemy_plane, plane.follow_plane)
                                # if TSVector3.distance(follow_enemy.pos3d, plane.pos3d)<self.attack_distance-1000 or len(plane.ready_attack):
                                if len(plane.ready_attack) and self.sim_time-plane.jammed>6:
                                    can_jammed_enemy[plane.ID] = jam_plane.pi_bound(TSVector3.calheading(TSVector3.minus(plane.pos3d,jam_plane.pos3d))-jam_plane.Heading)
                        can_jammed_enemy = [key for key, value in sorted(can_jammed_enemy.items(), key=lambda d: d[1])]
                    
                    if len(can_jammed_enemy):
                        if myjam:
                            middle_enemy = self.get_body_info_by_id(self.my_plane,can_jammed_enemy[int(len(can_jammed_enemy)/2)])
                            if jam_plane.can_see(middle_enemy,see_factor=0.6,jam_dis=117600):
                                jam_plane.do_turn = False
                                cmd_list.append(env_cmd.make_jamparam(jam_plane.ID))
                                jam_plane.middle_enemy_plane = None
                            else:
                                if jam_plane.ID in self.free_plane:
                                    self.free_plane.remove(jam_plane.ID)
                                if jam_plane.move_order==None:
                                    jam_plane.do_turn = True
                                    jam_plane.total_threat_flag = None
                                    jam_plane.move_order="搜寻干扰1"
                                    cmd_list.append(env_cmd.make_followparam(jam_plane.ID, middle_enemy.ID, jam_plane.move_speed, jam_plane.para['move_max_acc'], jam_plane.para['move_max_g']))
                        else:
                            jam_plane.middle_enemy_plane = can_jammed_enemy[int(len(can_jammed_enemy)/2)]
                            # 万一有人机遇到导弹威胁时middle有
                            middle_enemy = self.get_body_info_by_id(self.enemy_plane,jam_plane.middle_enemy_plane)
                            if jam_plane.can_see(middle_enemy,see_factor=0.6,jam_dis=117600):
                                jam_plane.do_turn = False
                                cmd_list.append(env_cmd.make_jamparam(jam_plane.ID))
                                jam_plane.middle_enemy_plane = None
                            else:
                                jam_enemy = self.get_body_info_by_id(self.enemy_plane, jam_plane.middle_enemy_plane)
                                turn_time = self.get_turn_time(jam_plane,jam_enemy,see_factor=0.6)
                                can_jam_after_turn = False
                                for missile_id in jammed_missile_list:
                                    missile = self.get_body_info_by_id(self.missile_list, missile_id)
                                    if missile.arrive_time>=turn_time+3:
                                        can_jam_after_turn = True
                                if can_jam_after_turn and len(jam_plane.locked_missile_list)==0:
                                    if jam_plane.ID in self.free_plane:
                                        self.free_plane.remove(jam_plane.ID)
                                    self.find_way_jam(jam_plane, jam_plane.middle_enemy_plane, cmd_list)

             # 无人机生存之道模块      
            # 无人机脱离近距离攻击范围
            for plane in self.my_plane:
                if plane.ID in self.free_plane and plane.Type==2:
                    if plane.ready_missile:
                        if plane.be_seen_leader:
                            total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(plane, see_dis=22000,seen=False)
                        else:
                            total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(plane, see_dis=12000,seen=False)
                    else:
                        total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(plane, see_dis=25000,seen=False)
                    if total_dir != {"X": 0, "Y": 0, "Z": 0}:
                        if min_dis<16000:
                            plane.be_in_danger = True
                            if plane.wing_who:
                                be_wing_plane = self.get_body_info_by_id(self.my_plane, plane.wing_who)
                                be_wing_plane.wing_plane = None
                                plane.wing_who = None
                        if plane.be_seen_myleader and plane.be_seen_leader==False and plane.ready_missile:
                            continue
                        if self.is_in_center(plane,140000) == False:
                            tmp_dir2 = {"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000}
                            tmp_dir2 = TSVector3.minus(tmp_dir2, plane.pos3d)
                            total_dir = tmp_dir2
                            min_dis = 9999999
                        self.free_plane.remove(plane.ID)
                        if plane.ID in no_missile_plane:
                            no_missile_plane.remove(plane.ID)
                        total_heading = TSVector3.calheading(total_dir)
                        total_pitch = -1*threat_pitch
                        if min_dis>14000 and min_dis != 99999999:
                            turn_ratio = 1/2
                            if self.is_in_center(plane,110000)==False:
                                turn_ratio = 4/9
                            # 转弯方向需要重新判断
                            if plane.total_threat_flag == None:
                                total_threat_dir,  min_dis2, threat_pitch = self.synthetic_threat_vector(plane,see_dis=30000,seen=False)
                                total_pitch = -1*threat_pitch
                                total_threat_heading = TSVector3.calheading(total_threat_dir)
                                if plane.pi_bound(total_heading-total_threat_heading)<0:
                                    total_heading = total_heading - math.pi*turn_ratio
                                    plane.total_threat_flag = -1
                                else:
                                    total_heading = total_heading + math.pi*turn_ratio
                                    plane.total_threat_flag = 1
                            else:
                                total_heading = total_heading + plane.total_threat_flag*math.pi*turn_ratio
                        elif min_dis<7000:
                            total_heading = plane.Heading
                            if self.sim_time%2:
                                total_pitch = -math.pi/6
                            else:
                                total_pitch = math.pi/6      
                        total_dir = TSVector3.calorientation(total_heading, total_pitch)
                        total_dir = TSVector3.normalize(total_dir)
                        distance = 20 * plane.move_speed
                        evade_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(total_dir, distance))
                        if evade_pos['Z'] < 2000:
                            evade_pos['Z'] = plane.Z+math.sin(abs(total_pitch))*distance
                        elif evade_pos['Z'] > plane.para["area_max_alt"]:
                            evade_pos['Z'] = plane.Z-math.sin(abs(total_pitch))*distance
                        straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": evade_pos['Z']}, ]
                        if plane.move_order==None:
                            plane.move_order="无人机生存之道"
                            cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, straight_evade_route_list,
                                            plane.move_speed, plane.para["move_max_acc"], plane.para["move_max_g"]))
                        else:
                            print(plane.ID,plane.move_order,"无人机生存之道")    

            # 给己方被威胁飞机安排跟踪飞机
            for plane in self.my_plane:
                if plane.be_in_danger and plane.Availability and plane.wing_plane==None:
                    find_wing_plane = None
                    for other_plane in self.my_plane:
                        if other_plane.ID != plane.ID and other_plane.be_in_danger==False and other_plane.ID in self.free_plane and other_plane.move_order==None and other_plane.wing_who==None:
                            if other_plane.can_see(plane) and find_wing_plane and other_plane.threat_ratio<find_wing_plane.threat_ratio:
                                find_wing_plane = other_plane
                            elif other_plane.can_see(plane) and find_wing_plane==None:
                                find_wing_plane = other_plane
                    if find_wing_plane==None:
                        for other_plane in self.my_plane:
                            if other_plane.ID != plane.ID and other_plane.be_in_danger==False and other_plane.ID in self.free_plane and other_plane.move_order==None and other_plane.wing_who==None:
                                if TSVector3.distance(other_plane.pos3d, plane.pos3d)<other_plane.para['radar_range'] and \
                                    find_wing_plane and self.get_turn_time(other_plane,plane)<self.get_turn_time(find_wing_plane,plane):
                                    find_wing_plane = other_plane
                                elif TSVector3.distance(other_plane.pos3d, plane.pos3d)<other_plane.para['radar_range'] and find_wing_plane==None:
                                    find_wing_plane = other_plane
                    if find_wing_plane==None:
                        for other_plane in self.my_plane:
                            if other_plane.ID != plane.ID and other_plane.be_in_danger==False and other_plane.ID in self.free_plane and other_plane.move_order==None and other_plane.wing_who==None:
                                if find_wing_plane and TSVector3.distance(other_plane.pos3d, plane.pos3d)< TSVector3.distance(find_wing_plane.pos3d, plane.pos3d):
                                    find_wing_plane = other_plane
                                elif find_wing_plane==None:
                                    find_wing_plane = other_plane
                    if find_wing_plane:
                        plane.wing_plane = find_wing_plane.ID
                        find_wing_plane.wing_who = plane.ID
                        self.free_plane.remove(find_wing_plane.ID)
                        find_wing_plane.move_order = "拯救大兵"
                        cmd_list.append(env_cmd.make_followparam(find_wing_plane.ID, plane.ID, find_wing_plane.move_speed, find_wing_plane.para['move_max_acc'], find_wing_plane.para['move_max_g']))

            # 指定僚机
            if len(self.enemy_plane) and self.sim_time>6*60:
                for leader_plane in self.my_leader_plane:
                    if leader_plane.Availability:
                        if leader_plane.wing_plane != None:
                            wing_plane = self.get_body_info_by_id(self.my_plane, leader_plane.wing_plane)
                            if (wing_plane.Type == 1 and self.revenge) or wing_plane.Type==2:
                                if wing_plane.Availability and len(leader_plane.locked_missile_list):
                                    if wing_plane.ID in self.free_plane:
                                        for plane_id in self.free_plane:
                                            plane = self.get_body_info_by_id(self.my_plane, plane_id)
                                            if plane.can_see(leader_plane,see_factor=0.9) and plane.threat_ratio<wing_plane.threat_ratio:
                                                wing_plane.wing_who = None
                                                wing_plane = plane
                                                leader_plane.wing_plane = plane.ID
                                                wing_plane.wing_who = leader_plane.ID
                                        self.free_plane.remove(wing_plane.ID)
                                        continue
                                    else:
                                        for plane_id in self.free_plane:
                                            plane = self.get_body_info_by_id(self.my_plane, plane_id)
                                            if plane.can_see(leader_plane,see_factor=0.9):
                                                wing_plane.wing_who = None
                                                leader_plane.wing_plane = None
                                                break
                                        if wing_plane.wing_who!=None:
                                            continue
                                if wing_plane.ID not in self.free_plane:
                                    wing_plane.wing_who = None
                                    leader_plane.wing_plane = None
                                elif wing_plane.ready_missile and len(no_missile_plane)>0:
                                    for plane_id in no_missile_plane:
                                        plane = self.get_body_info_by_id(self.my_plane, plane_id)
                                        if plane.can_see(leader_plane,see_factor=0.99):
                                            wing_plane.wing_who = None
                                            leader_plane.wing_plane = None  
                                            break
                                if leader_plane.wing_plane and wing_plane.ID in self.free_plane:
                                    self.free_plane.remove(wing_plane.ID)
                            else:
                                leader_plane.wing_plane = None
                                wing_plane.wing_who = None
                        if leader_plane.wing_plane == None:
                            for plane_id in no_missile_plane:
                                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                                if leader_plane.wing_plane:
                                    wing_plane = self.get_body_info_by_id(self.my_plane, leader_plane.wing_plane)
                                    if plane.threat_ratio < wing_plane.threat_ratio:
                                        if (plane.Type==2 or self.revenge) and plane.wing_who==None:
                                            wing_plane.wing_who = None
                                            leader_plane.wing_plane = plane_id
                                elif (plane.Type==2 or self.revenge) and plane.wing_who==None:
                                    leader_plane.wing_plane = plane_id
                                    # dis = TSVector3.distance(plane.pos3d, leader_plane.pos3d)
                            if leader_plane.wing_plane:
                                self.get_body_info_by_id(self.my_plane, leader_plane.wing_plane).wing_who = leader_plane.ID
                                no_missile_plane.remove(leader_plane.wing_plane)
                                if leader_plane.wing_plane in self.free_plane:
                                    self.free_plane.remove(leader_plane.wing_plane)
                            else:
                                have_missile = 3
                                for plane_id in self.free_plane:
                                    plane = self.get_body_info_by_id(self.my_plane, plane_id)
                                    if leader_plane.wing_plane:
                                        wing_plane = self.get_body_info_by_id(self.my_plane, leader_plane.wing_plane)
                                        if plane.threat_ratio < wing_plane.threat_ratio:
                                            if plane.Type == 2 and plane.ready_missile <= have_missile and plane.wing_who==None:
                                                wing_plane.wing_who = None
                                                have_missile = plane.ready_missile
                                                leader_plane.wing_plane = plane_id
                                    elif plane.Type == 2 and plane.ready_missile <= have_missile and plane.wing_who==None:
                                        have_missile = plane.ready_missile
                                        leader_plane.wing_plane = plane_id
                                if leader_plane.wing_plane is not None:
                                    self.get_body_info_by_id(self.my_plane, leader_plane.wing_plane).wing_who = leader_plane.ID
                                    self.free_plane.remove(leader_plane.wing_plane) 
                    else:
                        if leader_plane.wing_plane:
                            wing_plane = self.get_body_info_by_id(self.my_plane, leader_plane.wing_plane)
                            wing_plane.wing_who = None
                            leader_plane.wing_plane = None

            # 是否存在掉头攻击的可能
            for plane in self.my_plane:
                if plane.Availability and plane.ready_missile and plane.move_order==None and plane.do_turn==False and plane.do_jam==False and plane.ID in self.free_plane:
                    enemy_have_missile = False
                    for enemy in self.enemy_plane:
                        if enemy.Availability==0 and enemy.can_see(plane, jam_dis=plane.para["safe_range"]+1500) and enemy.ready_missile:
                            enemy_have_missile = True
                    if enemy_have_missile==False or (plane.Type==1 and self.sim_time-plane.last_jam>60) or (plane.wing_plane==None) or plane.be_seen_myleader:
                        dis = 9999999
                        turn_attack_plane = None
                        for enemy in self.enemy_plane:
                            if enemy.lost_flag==0:
                                both_dis = TSVector3.distance(enemy.pos3d, plane.pos3d)
                                if both_dis <self.attack_distance-500 and both_dis<dis:
                                    dis = both_dis
                                    turn_attack_plane = enemy
                        if turn_attack_plane:
                            plane.total_threat_flag = None
                            self.free_plane.remove(plane.ID)
                            if plane.ID in no_missile_plane:
                                no_missile_plane.remove(plane.ID)
                            plane.move_order = "掉头攻击"
                            cmd_list.append(env_cmd.make_followparam(plane.ID, turn_attack_plane.ID, plane.move_speed, plane.para['move_max_acc'], plane.para['move_max_g']))
                        
            # 有人机脱离战场
            for leader in self.my_leader_plane:
                need_wing = False
                if leader.ready_missile:
                    total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(leader, seen=False)
                else:
                    total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(leader, see_dis=26000, seen=False)
                if self.enemy_left_weapon==0:
                    break
                if leader.ID in self.free_plane:
                    if total_dir != {"X": 0, "Y": 0, "Z": 0} and leader.do_jam==False:
                        if min_dis<16000:
                            leader.be_in_danger = True
                        next_heading = TSVector3.calheading(total_dir) + math.pi
                        if self.my_score>self.enemy_score or self.enemy_strategy() or self.sim_time<12*60:# self.enemy_formation()==False:
                            center_radius = 140000
                        elif self.my_score==self.enemy_score and (self.my_center_time<self.enemy_center_time*2 or self.my_center_time<self.enemy_center_time+(20*60-self.sim_time)*self.live_enemy_leader):
                            center_radius = 45000
                        else:
                            center_radius = 100000
                        if self.is_in_center(leader,center_radius) == False:
                            tmp_dir2 = {"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000}
                            tmp_dir2 = TSVector3.minus(tmp_dir2, leader.pos3d)
                            total_dir = tmp_dir2
                            min_dis = 9999999
                        self.free_plane.remove(leader.ID)
                        total_heading = TSVector3.calheading(total_dir)
                        total_pitch = -1*threat_pitch
                        if min_dis>26000 and min_dis != 99999999:
                            turn_ratio = 1/2
                            if self.is_in_center(leader,center_radius+5000)==False:
                                turn_ratio = 4/9
                            # 转弯方向需要重新判断
                            if leader.total_threat_flag==None:
                                total_threat_dir,  min_dis2, threat_pitch = self.synthetic_threat_vector(leader,see_dis=35000,seen=False)
                                total_pitch = -1*threat_pitch
                                total_threat_heading = TSVector3.calheading(total_threat_dir)
                                if leader.pi_bound(total_heading-total_threat_heading)<0:
                                    total_heading = total_heading - math.pi*turn_ratio
                                    leader.total_threat_flag = -1
                                else:
                                    total_heading = total_heading + math.pi*turn_ratio
                                    leader.total_threat_flag = 1
                            else:
                                total_heading = total_heading + leader.total_threat_flag*math.pi*turn_ratio
                        elif min_dis<8000:
                            total_heading = plane.Heading
                            if self.sim_time%2:
                                total_pitch = -math.pi/6
                            else:
                                total_pitch = math.pi/6
                            
                        total_dir = TSVector3.calorientation(total_heading, total_pitch)
                        total_dir = TSVector3.normalize(total_dir)
                        distance = 20 * leader.move_speed
                        evade_pos = TSVector3.plus(leader.pos3d, TSVector3.multscalar(total_dir, distance))
                        if evade_pos['Z'] < 2000:
                            evade_pos['Z'] = leader.Z+math.sin(abs(total_pitch))*distance
                        elif evade_pos['Z'] > leader.para["area_max_alt"]:
                            evade_pos['Z'] = leader.Z-math.sin(abs(total_pitch))*distance
                        straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": evade_pos['Z']}, ]
                        if leader.move_order==None:
                            leader.move_order="leader规避风险"
                            cmd_list.append(env_cmd.make_linepatrolparam(leader.ID, straight_evade_route_list,
                                            leader.move_speed, leader.para["move_max_acc"], leader.para["move_max_g"]))
                        else:
                            print(leader.ID,leader.move_order,"leader规避风险")
                        if leader.Availability and leader.wing_plane:
                            wing_plane = self.get_body_info_by_id(self.my_plane, leader.wing_plane)
                            if wing_plane.move_order==None:
                                if wing_plane.ID in self.free_plane:
                                    self.free_plane.remove(wing_plane.ID)
                                wing_plane.move_order="僚机35000"
                                wing_plane.total_threat_flag = None
                                dis = TSVector3.distance(wing_plane.pos3d, leader.pos3d)
                                if dis>35000:
                                    wing_plane.move_speed = wing_plane.para["move_max_speed"]
                                    cmd_list.append(env_cmd.make_followparam(wing_plane.ID, leader.ID, wing_plane.move_speed, wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
                                else:
                                    next_pitch = 0
                                    enemy_sum = 0
                                    for enemy in self.enemy_plane:
                                        if enemy.Availability:
                                            if abs(wing_plane.pi_bound(TSVector3.calheading(TSVector3.minus(enemy.pos3d, wing_plane.pos3d))-next_heading))>wing_plane.para['radar_heading']/180*math.pi/2:
                                                if TSVector3.distance(enemy.pos3d, wing_plane.pos3d)<wing_plane.para['radar_range']:
                                                    next_pitch += TSVector3.calpitch(TSVector3.minus(enemy.pos3d, wing_plane.pos3d))*15000/(TSVector3.distance(enemy.pos3d, wing_plane.pos3d)+200)
                                                    enemy_sum += 15000/(TSVector3.distance(enemy.pos3d, wing_plane.pos3d)+200)
                                    if enemy_sum>0:
                                        next_pitch /= enemy_sum
                                    total_dir = TSVector3.calorientation(next_heading, next_pitch)
                                    total_dir = TSVector3.normalize(total_dir)
                                    distance = 20 * wing_plane.para["move_max_speed"]
                                    next_pos = TSVector3.plus(wing_plane.pos3d, TSVector3.multscalar(total_dir, distance))
                                    next_pos['Z'] = leader.Z
                                    if wing_plane.move_speed > leader.Speed:
                                        wing_plane.move_speed = leader.Speed
                                    cmd_list.append(env_cmd.make_linepatrolparam(wing_plane.ID, [next_pos,],
                                                wing_plane.move_speed, wing_plane.para["move_max_acc"], wing_plane.para["move_max_g"]))
                            else:
                                print(leader.ID,wing_plane.ID,wing_plane.move_order,"僚机35000")
                            
                    else:
                        # 当没有威胁时，主动去找敌方
                        support_uav_plane_list = {}
                        for plane in self.my_plane:
                            if plane.ID != leader.ID and plane.Availability:
                                if plane.be_in_danger and plane.ID not in support_uav_plane_list or plane.threat_ratio:
                                    dis =  leader.pi_bound(TSVector3.calheading(TSVector3.minus(plane.pos3d,leader.pos3d))-leader.Heading)
                                    support_uav_plane_list[plane.ID] = dis
                        if len(support_uav_plane_list):
                            support_uav_plane_list = [key for key, value in sorted(support_uav_plane_list.items(), key=lambda d: d[1])]
                            middile_my_plane = support_uav_plane_list[int(len(support_uav_plane_list)/2)]
                            if leader.move_order==None:
                                leader.move_order = "提前为无人机提供隐身视野支持"
                                self.free_plane.remove(leader.ID)
                                leader.total_threat_flag = None
                                cmd_list.append(env_cmd.make_followparam(leader.ID, middile_my_plane, leader.move_speed, leader.para['move_max_acc'], leader.para['move_max_g']))
                            else:
                                print(leader.ID,leader.move_order,"提前为无人机提供隐身视野支持")
                        elif len(self.enemy_plane):
                            detect_enemy_list = {}
                            for enemy in self.enemy_plane:
                                if self.enemy_left_weapon>0:
                                    if enemy.Availability and enemy.ready_missile and enemy.ID not in detect_enemy_list and TSVector3.distance(enemy.pos3d,leader.pos3d)<leader.para['radar_range']*0.9:
                                        dis =  leader.pi_bound(TSVector3.calheading(TSVector3.minus(enemy.pos3d,leader.pos3d))-leader.Heading)
                                        detect_enemy_list[enemy.ID] = dis
                                else:
                                    if enemy.Availability and enemy.ID not in detect_enemy_list:
                                        dis =  leader.pi_bound(TSVector3.calheading(TSVector3.minus(enemy.pos3d,leader.pos3d))-leader.Heading)
                                        detect_enemy_list[enemy.ID] = dis
                            if len(detect_enemy_list):
                                detect_enemy_list = [key for key, value in sorted(detect_enemy_list.items(), key=lambda d: d[1])]
                                middile_enemy_plane_id = detect_enemy_list[int(len(detect_enemy_list)/2)]
                                middile_enemy_plane = self.get_body_info_by_id(self.enemy_plane,middile_enemy_plane_id)
                                if leader.move_order==None:
                                    self.free_plane.remove(leader.ID)
                                    leader.move_order = "找到敌方飞机视野"
                                    leader.total_threat_flag = None
                                    dis = TSVector3.distance(leader.pos3d, middile_enemy_plane.pos3d)
                                    if leader.ready_missile:
                                        far_away_dis = self.attack_distance
                                    else:
                                        far_away_dis = leader.para["safe_range"]
                                    if dis<far_away_dis:
                                        if leader.move_speed > middile_enemy_plane.Speed:
                                            leader.move_speed = max(middile_enemy_plane.Speed-5,leader.para["move_min_speed"])
                                    if middile_enemy_plane.lost_flag==0:
                                        cmd_list.append(env_cmd.make_followparam(leader.ID, middile_enemy_plane.ID, leader.move_speed, leader.para['move_max_acc'], leader.para['move_max_g']))
                                    else:
                                        cmd_list.append(env_cmd.make_linepatrolparam(leader.ID, [middile_enemy_plane.pos3d,],
                                            leader.move_speed, leader.para["move_max_acc"], leader.para["move_max_g"]))
                                else:
                                    print(leader.ID,leader.move_order,"找到敌方飞机视野")
                        need_wing = True
                else:
                    need_wing = True
                if leader.Availability and need_wing:
                    if leader.wing_plane:
                        wing_plane = self.get_body_info_by_id(self.my_plane, leader.wing_plane)
                        if wing_plane.close_missile!=None and TSVector3.distance(wing_plane.pos3d,wing_plane.close_missile.pos3d)<10000:
                            continue
                        if len(leader.locked_missile_list) > 0:
                            if wing_plane.move_order==None:
                                if wing_plane.ID in self.free_plane:
                                    self.free_plane.remove(wing_plane.ID)
                                follow_missile = leader.close_missile
                                wing_plane.move_order="无人机跟踪导弹00"
                                wing_plane.total_threat_flag = None
                                if follow_missile != None and follow_missile.lost_flag==0:
                                    cmd_list.append(env_cmd.make_followparam(wing_plane.ID, follow_missile.ID, wing_plane.move_speed, wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
                                elif follow_missile != None and follow_missile.lost_flag>0:
                                    have_seen_missile = False
                                    for missile_id in leader.locked_missile_list:
                                        missile =  self.get_body_info_by_id(self.missile_list, missile_id)
                                        if missile.lost_flag==0:
                                            have_seen_missile = True
                                            cmd_list.append(env_cmd.make_followparam(wing_plane.ID, missile.ID, wing_plane.move_speed, wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
                                            break
                                    if have_seen_missile==False:
                                        total_dir = TSVector3.calorientation(follow_missile.Heading, follow_missile.Pitch)
                                        distance = follow_missile.Speed
                                        next_pos = TSVector3.plus(follow_missile.pos3d, TSVector3.multscalar(total_dir, distance))
                                        cmd_list.append(env_cmd.make_linepatrolparam(wing_plane.ID, [next_pos,],
                                                    wing_plane.move_speed, wing_plane.para["move_max_acc"], wing_plane.para["move_max_g"]))
                                else:
                                    if TSVector3.distance(wing_plane.pos3d, leader.pos3d)>35000:
                                        wing_plane.move_speed = wing_plane.para['move_max_speed']
                                    else:
                                        if wing_plane.move_speed > leader.Speed:
                                            wing_plane.move_speed = leader.Speed
                                    cmd_list.append(env_cmd.make_followparam(wing_plane.ID, leader.ID, wing_plane.move_speed, wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
                            else:
                                print(wing_plane.ID,wing_plane.move_order,"无人机跟踪导弹00")
                            
                        else:
                            if wing_plane.move_order==None:
                                if wing_plane.ID in self.free_plane:
                                    self.free_plane.remove(wing_plane.ID)
                                wing_plane.move_order="僚机leader"
                                dis = TSVector3.distance(wing_plane.pos3d, leader.pos3d)
                                wing_plane.total_threat_dir = None
                                if dis>35000:
                                    wing_plane.move_speed = wing_plane.para['move_max_speed']
                                    cmd_list.append(env_cmd.make_followparam(wing_plane.ID, leader.ID, wing_plane.move_speed, wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
                                elif dis<30000:
                                    if wing_plane.move_speed > leader.Speed:
                                        wing_plane.move_speed = leader.Speed
                                    next_heading = leader.Heading + math.pi
                                    if next_heading>2*math.pi:
                                        next_heading -= 2*math.pi
                                    next_pitch = 0
                                    enemy_sum = 0
                                    for enemy in self.enemy_plane:
                                        if enemy.Availability:
                                            if abs(wing_plane.pi_bound(TSVector3.calheading(TSVector3.minus(enemy.pos3d, wing_plane.pos3d))-next_heading))>wing_plane.para['radar_heading']/180*math.pi/2:
                                                if TSVector3.distance(enemy.pos3d, wing_plane.pos3d)<wing_plane.para['radar_range']:
                                                    next_pitch += TSVector3.calpitch(TSVector3.minus(enemy.pos3d, wing_plane.pos3d))*15000/(TSVector3.distance(enemy.pos3d, wing_plane.pos3d)+200)
                                                    enemy_sum += 15000/(TSVector3.distance(enemy.pos3d, wing_plane.pos3d)+200)
                                    if enemy_sum>0:
                                        next_pitch /= enemy_sum
                                    total_dir = TSVector3.calorientation(next_heading, next_pitch)
                                    total_dir = TSVector3.normalize(total_dir)
                                    distance = 20 * wing_plane.para["move_max_speed"]
                                    next_pos = TSVector3.plus(wing_plane.pos3d, TSVector3.multscalar(total_dir, distance))
                                    cmd_list.append(env_cmd.make_linepatrolparam(wing_plane.ID, [next_pos,],
                                                wing_plane.move_speed, wing_plane.para["move_max_acc"], wing_plane.para["move_max_g"]))
                            else:
                                print(leader.ID,wing_plane.ID,wing_plane.move_order,"僚机leader")           
            # 扩大视野模块
            # 在跟踪其他飞机
            for my_plane in self.my_plane:
                if my_plane.ID in self.free_plane and my_plane.ready_missile:
                    if my_plane.follow_plane:
                        enemy = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                        if my_plane.ID not in enemy.followed_plane:
                            my_plane.follow_plane = None
                    if my_plane.follow_plane == None:
                        dis = 999999999
                        for enemy_plane in self.enemy_plane:
                            if enemy_plane.lost_flag:
                                if TSVector3.distance(enemy_plane.pos3d, my_plane.pos3d) < dis and len(enemy_plane.followed_plane)<2:
                                    dis = TSVector3.distance(enemy_plane.pos3d, my_plane.pos3d)
                                    my_plane.follow_plane = enemy_plane.ID
                        if my_plane.follow_plane is not None and enemy_plane.lost_flag<self.get_turn_time(my_plane,enemy_plane)+2 and enemy_plane.Availability:
                            if my_plane.move_order==None:
                                enemy_plane = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                                if my_plane.ID not in enemy_plane.followed_plane:
                                    enemy_plane.followed_plane.append(my_plane.ID)
                                self.free_plane.remove(my_plane.ID)
                                my_plane.move_order="跟踪敌机"
                                my_plane.total_threat_flag = None
                                min_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                                total_heading = TSVector3.calheading(TSVector3.minus(enemy_plane.pos3d,my_plane.pos3d))
                                total_pitch = TSVector3.calpitch(TSVector3.minus(enemy_plane.pos3d,my_plane.pos3d))
                                if min_dis<9000 or my_plane.be_seen==False:
                                    if self.sim_time%2:
                                        total_pitch = -math.pi/6
                                    else:
                                        total_pitch = math.pi/6      
                                total_dir = TSVector3.calorientation(total_heading, total_pitch)
                                total_dir = TSVector3.normalize(total_dir)
                                distance = min_dis
                                evade_pos = TSVector3.plus(my_plane.pos3d, TSVector3.multscalar(total_dir, distance))
                                if evade_pos['Z'] < 2000:
                                    evade_pos['Z'] = my_plane.Z+math.sin(abs(total_pitch))*distance
                                elif evade_pos['Z'] > my_plane.para["area_max_alt"]:
                                    evade_pos['Z'] = my_plane.Z-math.sin(abs(total_pitch))*distance
                                straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": evade_pos['Z']}, ]
                                cmd_list.append(env_cmd.make_linepatrolparam(my_plane.ID, straight_evade_route_list,
                                                my_plane.para["move_max_speed"], my_plane.para["move_max_acc"], my_plane.para["move_max_g"]))
                            else:
                                my_plane.follow_plane = None
                                print(my_plane.ID,my_plane.move_order,"跟踪敌机")
                    elif my_plane.follow_plane and my_plane.move_order==None:
                        # 选择更近的飞机进行追踪
                        enemy_plane = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                        new_follow_plane = enemy_plane
                        if new_follow_plane.lost_flag>10:
                            enemy_plane.followed_plane.remove(my_plane.ID)
                            my_plane.follow_plane = None
                        for other_enemy_plane in self.enemy_plane: # 遇到更近的敌人转移
                            if new_follow_plane and other_enemy_plane.ID != new_follow_plane.ID:
                                if TSVector3.distance(my_plane.pos3d, new_follow_plane.pos3d) > TSVector3.distance(my_plane.pos3d, other_enemy_plane.pos3d) + 800 \
                                        and other_enemy_plane.lost_flag==0:
                                    if len(other_enemy_plane.followed_plane)>0:
                                        have_missile = False
                                        for follow_plane_id in other_enemy_plane.followed_plane:
                                            follow_plane = self.get_body_info_by_id(self.my_plane, follow_plane_id)
                                            if follow_plane.ready_missile>0 and TSVector3.distance(follow_plane.pos3d, other_enemy_plane.pos3d)<TSVector3.distance(my_plane.pos3d, other_enemy_plane.pos3d)+1000:
                                                have_missile = True
                                        if have_missile==False:
                                            new_follow_plane = other_enemy_plane
                                    else:
                                        new_follow_plane = other_enemy_plane
                            elif new_follow_plane==None and other_enemy_plane.lost_flag==0:
                                new_follow_plane = other_enemy_plane
                        if my_plane.ID in enemy_plane.followed_plane:
                            enemy_plane.followed_plane.remove(my_plane.ID)
                        if new_follow_plane:
                            new_follow_plane.followed_plane.append(my_plane.ID)
                            my_plane.follow_plane = new_follow_plane.ID
                            enemy_plane = new_follow_plane
                            min_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                            total_heading = TSVector3.calheading(TSVector3.minus(enemy_plane.pos3d,my_plane.pos3d))
                            total_pitch = TSVector3.calpitch(TSVector3.minus(enemy_plane.pos3d,my_plane.pos3d))
                            dou = False
                            if new_follow_plane.lost_flag>0 and len(new_follow_plane.lost_missile_list)==0:
                                dou = True
                            if min_dis<9000 or my_plane.be_seen==False or dou:
                                if self.sim_time%2:
                                    total_pitch = -math.pi/6
                                else:
                                    total_pitch = math.pi/6      
                            total_dir = TSVector3.calorientation(total_heading, total_pitch)
                            total_dir = TSVector3.normalize(total_dir)
                            distance = 20 * my_plane.move_speed
                            evade_pos = TSVector3.plus(my_plane.pos3d, TSVector3.multscalar(total_dir, distance))
                            if evade_pos['Z'] < 2000:
                                evade_pos['Z'] = my_plane.Z+math.sin(abs(total_pitch))*distance
                            elif evade_pos['Z'] > my_plane.para["area_max_alt"]:
                                evade_pos['Z'] = my_plane.Z-math.sin(abs(total_pitch))*distance
                            straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": evade_pos['Z']}, ]
                            if my_plane.move_order==None:
                                self.free_plane.remove(my_plane.ID)
                                cmd_list.append(env_cmd.make_linepatrolparam(my_plane.ID, straight_evade_route_list,
                                            my_plane.para["move_max_speed"], my_plane.para["move_max_acc"], my_plane.para["move_max_g"]))
                            else:
                                my_plane.follow_plane = None
                                print(my_plane.ID,my_plane.move_order,"跟踪敌机")
             # 飞机波浪式S型转圈视野扩展
            s_my_plane = sorted(self.my_plane, key=lambda d: d.ready_missile, reverse=False)
            if len(self.enemy_plane)>0:
                for plane in s_my_plane:
                    if plane.ID in self.free_plane:
                        plane.enemy_heading_range = [2*math.pi, 0]
                        plane.enemy_pitch_range = [math.pi/4, -math.pi/4]
                        can_see_enemy = False
                        for enemy in self.enemy_plane:
                            if TSVector3.distance(enemy.pos3d, plane.pos3d)<plane.para['radar_range'] and enemy.Availability:
                                enemy_heading = TSVector3.calheading(TSVector3.minus(enemy.pos3d, plane.pos3d))
                                enemy_pitch = TSVector3.calpitch(TSVector3.minus(enemy.pos3d, plane.pos3d))
                                can_see_enemy = True
                                if enemy_heading < plane.enemy_heading_range[0]:
                                    plane.enemy_heading_range[0] = enemy_heading+plane.para['radar_heading']*0.5/180*math.pi
                                if enemy_heading > plane.enemy_heading_range[1]:
                                    plane.enemy_heading_range[1] = enemy_heading-plane.para['radar_heading']*0.5/180*math.pi
                                if enemy_pitch < plane.enemy_pitch_range[0]:
                                    plane.enemy_pitch_range[0] = enemy_pitch+plane.para['radar_pitch']*0.5/180*math.pi
                                if enemy_pitch > plane.enemy_pitch_range[1]:
                                    plane.enemy_pitch_range[1] = enemy_pitch-plane.para['radar_pitch']*0.5/180*math.pi
                        if can_see_enemy==False:
                            for enemy in self.my_plane:
                                if TSVector3.distance(enemy.pos3d, plane.pos3d)<plane.para['radar_range'] and enemy.Availability and enemy.ID!=plane.ID and plane.threat_ratio:
                                    enemy_heading = TSVector3.calheading(TSVector3.minus(enemy.pos3d, plane.pos3d))
                                    enemy_pitch = TSVector3.calpitch(TSVector3.minus(enemy.pos3d, plane.pos3d))
                                    can_see_enemy = True
                                    if enemy_heading < plane.enemy_heading_range[0]:
                                        plane.enemy_heading_range[0] = enemy_heading+plane.para['radar_heading']*0.5/180*math.pi
                                    if enemy_heading > plane.enemy_heading_range[1]:
                                        plane.enemy_heading_range[1] = enemy_heading-plane.para['radar_heading']*0.5/180*math.pi
                                    if enemy_pitch < plane.enemy_pitch_range[0]:
                                        plane.enemy_pitch_range[0] = enemy_pitch+plane.para['radar_pitch']*0.5/180*math.pi
                                    if enemy_pitch > plane.enemy_pitch_range[1]:
                                        plane.enemy_pitch_range[1] = enemy_pitch-plane.para['radar_pitch']*0.5/180*math.pi
                        if can_see_enemy==False:
                            continue
                        if plane.Heading < plane.enemy_heading_range[0]:
                            plane.turn_heading_flag = 1
                        elif plane.Heading > plane.enemy_heading_range[1]:
                            plane.turn_heading_flag = -1
                        heading = (plane.Heading+plane.turn_heading_flag*45/180*math.pi)
                        if heading > 2*math.pi:
                            heading -= 2*math.pi
                        elif heading < 0:
                            heading += 2*math.pi
                        if self.sim_time%2:
                            total_pitch = -max(abs(plane.enemy_pitch_range[1]),abs(plane.enemy_pitch_range[0]))
                        else:
                            total_pitch = max(abs(plane.enemy_pitch_range[1]),abs(plane.enemy_pitch_range[0]))
                        total_dir = TSVector3.calorientation(heading, total_pitch)
                        distance = 15 * plane.Speed
                        evade_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(total_dir, distance))
                        vertical_evade_route_list = [evade_pos,]
                        if plane.move_order==None:
                            plane.move_order = "S转型圈波浪式扩展视野"
                            self.free_plane.remove(plane.ID)
                            cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, vertical_evade_route_list, plane.move_speed,
                                            plane.para["move_max_acc"], plane.para["move_max_g"]))
                        else:
                            print(plane.ID,plane.move_order,"S转型圈波浪式扩展视野")      
            # 无人机躲避模块
            for plane in self.my_uav_plane:
                if plane.close_missile == None or (TSVector3.distance(plane.close_missile.pos3d,plane.pos3d)>=11000 and plane.wing_who!=None):
                    continue
                total_dir,min_dis,threat_pitch = self.synthetic_threat_vector(plane,see_dis=45000,seen=False)
                new_plane_pos = plane.evade(plane.close_missile, total_dir, cmd_list)
                if self.death_to_death(plane, plane.close_missile, new_plane_pos):
                    self.all_death(plane, cmd_list)
            # 默认移动模块
            self.init_move(cmd_list)
            for plane in self.my_plane:
                if plane.wing_who!=None:
                    be_wing_plane = self.get_body_info_by_id(self.my_plane, plane.wing_who)
                    if be_wing_plane.wing_plane!=plane.ID:
                        print(plane.Name,'僚机不匹配')
                if plane.wing_plane!=None:
                    wing_plane = self.get_body_info_by_id(self.my_plane, plane.wing_plane)
                    if wing_plane.wing_who!=plane.ID:
                        print(plane.Name,"僚机不匹配")
        init_act = self.get_sector_index(0, 0)
        fixed_actions = [init_act for _ in range(10)]
        for cl in cmd_list:
            key = cl.keys()
            if "CmdLinePatrolControl" in key:
                plane_id = cl["CmdLinePatrolControl"]["Receiver"]
                plane_next_pos = cl["CmdLinePatrolControl"]["CoordList"][0]
                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                plane_now_pos = plane.pos3d
                heading = plane.pi_bound(TSVector3.calheading(TSVector3.minus(plane_next_pos, plane_now_pos)) - plane.Heading)
                pitch = TSVector3.calpitch(TSVector3.minus(plane_next_pos, plane_now_pos))
                act = self.get_sector_index(heading, pitch)
                fixed_actions[plane.i] = act
            elif "CmdTargetFollowControl" in key:
                plane_id = cl["CmdTargetFollowControl"]["Receiver"]
                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                fixed_actions[plane.i] = self.num_azimuth*self.num_elevation+1
            elif "CmdAttackControl" in key:
                plane_id = cl["CmdAttackControl"]["Receiver"]
                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                fixed_actions[plane.i] = self.num_azimuth*self.num_elevation
            elif "CmdJamControl" in key:
                plane_id = cl["CmdJamControl"]["Receiver"]
                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                fixed_actions[plane.i] = self.num_azimuth*self.num_elevation+2
        return fixed_actions, cmd_list
    # 获取相对导弹信息
    def get_relative_missile_info(self, plane, missile):
        Relative_X_i = np.array([0])
        Relative_Y_i = np.array([0])
        Relative_Z_i = np.array([0])
        Relative_dist_i = np.array([0])
        Relative_theta_i = np.array([0])
        Relative_alpha_i = np.array([0])
        loss_target = np.array([0])
        arrive_time = np.array([0])
        lost_flag = np.array([0])
        speed = np.array([0])
        missile_acc = np.array([0])
        heading = np.array([0])
        pitch = np.array([0])
        if missile is not None and plane is not None:
            Relative_X_i = np.array([missile.X - plane.X])
            Relative_Y_i = np.array([missile.Y - plane.Y])
            Relative_Z_i = np.array([missile.Z - plane.Z])
            Relative_dist_i = np.array([TSVector3.distance(plane.pos3d, missile.pos3d)])
            Relative_theta_i = np.array([plane.pi_bound(plane.XY2theta(Relative_X_i[0], Relative_Y_i[0]))])
            Relative_alpha_i = np.array([plane.pi_bound(math.asin(Relative_Z_i[0] / (Relative_dist_i[0] + self.EPS)))])
            loss_target = np.array([missile.loss_target])
            arrive_time = np.array([missile.arrive_time])
            if missile.lost_flag:
                lost_flag = np.array([1])
            speed = np.array([missile.Speed])
            missile_acc = np.array([missile.missile_acc])
            heading = np.array([missile.Heading])
            pitch = np.array([missile.Pitch])
            # 归一化处理
            Relative_X_i[0] /= 300000  # Relative_X_i:[-1, 1]
            Relative_Y_i[0] /= 300000  # Relative_Y_i:[-1, 1]
            Relative_Z_i[0] /= 13000  # Relative_Z_i:[-1, 1]
            Relative_dist_i[0] /= 424436  # Relative_dist_i:[0, 1]
            Relative_theta_i[0] /= math.pi  # Relative_theta_i:[-1, 1]
            Relative_alpha_i[0] /= (math.pi / 2)  # Relative_alpha_i:[-1, 1]
            arrive_time[0] /= 424436/1000
            speed[0] /= 100
            missile_acc[0] /= 98
            heading[0] /= math.pi*2
            pitch[0] /= (math.pi / 2)

        return np.concatenate((Relative_X_i,Relative_Y_i,Relative_Z_i,Relative_dist_i,
                                Relative_theta_i,Relative_alpha_i,loss_target,arrive_time,
                                lost_flag, speed, missile_acc, heading, pitch))
    
    # 获取个体obs
    def get_current_control_info(self, cur_plane):
        Identification_my = np.array([0]) if cur_plane.Identification == "红方" else np.array([1])
        Type_my = np.array([0]) if cur_plane.Type == 1 else np.array([1])
        Availability_my = np.array([cur_plane.Availability])
        X_my = np.array([0])
        Y_my = np.array([0])
        Z_my = np.array([0])
        Heading_my = np.array([0])
        Pitch_my = np.array([0])
        Roll_my = np.array([0])
        Speed_my = np.array([0])
        close_missile_my = self.get_relative_missile_info(cur_plane, cur_plane.close_missile)
        center_time = np.array([0])
        LeftWeapon_my = np.array([0])
        do_jam = np.array([0])
        jammed = np.array([0])
        be_seen_enemyleader = np.array([0])
        be_seen_myleader = np.array([0])
        be_seen = np.array([0])
        threat_ratio = np.array([0])
        if Availability_my:
            X_my = np.array([cur_plane.X])
            Y_my = np.array([cur_plane.Y])
            Z_my = np.array([cur_plane.Z])
            Heading_my = np.array([cur_plane.Heading])
            Pitch_my = np.array([cur_plane.Pitch])
            Roll_my = np.array([cur_plane.Roll])
            Speed_my = np.array([cur_plane.Speed])
            center_time = np.array([cur_plane.center_time])
            LeftWeapon_my = np.array([cur_plane.ready_missile])
            do_jam = np.array([cur_plane.do_jam])
            jammed = np.array([self.sim_time- 1 - cur_plane.jammed])
            be_seen_enemyleader = np.array([cur_plane.be_seen_leader])
            be_seen_myleader = np.array([cur_plane.be_seen_myleader])
            be_seen = np.array([cur_plane.be_seen])
            threat_ratio = np.array([cur_plane.threat_ratio])
        current_control_info = np.concatenate((Identification_my, Type_my, Availability_my, X_my,
                                               Y_my, Z_my, Heading_my, Pitch_my, Roll_my, Speed_my,
                                               close_missile_my, center_time, do_jam, LeftWeapon_my,
                                               jammed, be_seen_enemyleader, be_seen_myleader, be_seen,
                                                threat_ratio))
        # 归一化处理
        current_control_info[3] /= 150000  # X_my:[-1, 1]
        current_control_info[4] /= 150000  # Y_my:[-1, 1]
        current_control_info[5] = (current_control_info[5] - 2000) / 13000 if cur_plane.Type == 1 else (current_control_info[5] - 2000) / 8000  # Z_my:[0, 1]
        current_control_info[6] /= 2*math.pi  # Heading_my:[-1, 1]
        current_control_info[7] /= (math.pi/2)  # Pitch_my:[-1, 1]
        current_control_info[8] /= math.pi  # Roll_my:[-1, 1]
        current_control_info[9] = (current_control_info[9] - 150) / 250 if cur_plane.Type == 1 else (current_control_info[9] - 100) / 200  # Speed:[0, 1]
        current_control_info[11] /= 10*60  # center_time:[0, 1]
        current_control_info[13] /= (4 if cur_plane.Type == 1 else 2)  # LeftWeapon_my:[0, 1]
        current_control_info[14] = 7/(current_control_info[14]+1) # jammed:[0, 1]
        current_control_info[18] /= 1000  # threat_ratio:[0, 1]
        global_obs = self.get_global_obs()
        current_control_info = np.concatenate((current_control_info, self.get_enemy_allplane_info(cur_plane), global_obs))
        return current_control_info

    def get_enemy_allplane_info(self, cur_plane):
        enemy_allplane_info = []
        if len(self.enemy_plane):
            for plane in self.enemy_plane:
                Relative_X_i = np.array([0])
                Relative_Y_i = np.array([0])
                Relative_Z_i = np.array([0])
                Relative_dist_i = np.array([0])
                Relative_theta_i = np.array([0])
                Relative_alpha_i = np.array([0])
                heading = np.array([0])
                pitch = np.array([0])
                speed = np.array([0])
                ready_missile = np.array([0])
                Availability = np.array([0])
                lost_flag = np.array([0])
                jammed = np.array([0])
                be_seen_enemyleader = np.array([0])
                be_seen_myleader = np.array([0])
                be_seen = np.array([0])
                if plane.Availability and cur_plane is not None:
                    Relative_X_i = np.array([plane.X - cur_plane.X])
                    Relative_Y_i = np.array([plane.Y - cur_plane.Y])
                    Relative_Z_i = np.array([plane.Z - cur_plane.Z])
                    Relative_dist_i = np.array([TSVector3.distance(plane.pos3d, cur_plane.pos3d)])
                    Relative_theta_i = np.array([plane.pi_bound(plane.XY2theta(Relative_X_i[0], Relative_Y_i[0]))])
                    Relative_alpha_i = np.array([plane.pi_bound(math.asin(Relative_Z_i[0] / (Relative_dist_i[0] + self.EPS)))])
                    heading = np.array([plane.Heading])
                    pitch = np.array([plane.Pitch])
                    speed = np.array([plane.Speed])
                    ready_missile = np.array([plane.ready_missile])
                    Availability = np.array([plane.Availability])
                    if plane.lost_flag:
                        lost_flag = np.array([1])
                    # 归一化处理
                    Relative_X_i[0] /= 300000  # Relative_X_i:[-1, 1]
                    Relative_Y_i[0] /= 300000  # Relative_Y_i:[-1, 1]
                    Relative_Z_i[0] = (Relative_Z_i[0] - 2000) / 13000 if cur_plane.Type == 1 else (Relative_Z_i[0] - 2000) / 8000  # Relative_Z_i:[0, 1]
                    Relative_dist_i[0] /= 424436  # Relative_dist_i:[0, 1]
                    Relative_theta_i[0] /= 2*math.pi  # Relative_theta_i:[-1, 1]
                    Relative_alpha_i[0] /= (math.pi / 2)  # Relative_alpha_i:[-1, 1]
                    heading[0] /= math.pi*2
                    pitch[0] /= (math.pi / 2)
                    speed[0] = (speed[0] - 150) / 250 if cur_plane.Type == 1 else (speed[0] - 100) / 200  # Speed:[0, 1]
                    jammed = np.array([self.sim_time - cur_plane.jammed])
                    be_seen_enemyleader = np.array([cur_plane.be_seen_leader])
                    be_seen_myleader = np.array([cur_plane.be_seen_myleader])
                    be_seen = np.array([cur_plane.be_seen])
                    jammed = 7/(jammed+1)  # jammed:[0, 1]
                enemy_info = np.concatenate((Relative_X_i,Relative_Y_i,Relative_Z_i,Relative_dist_i,Relative_theta_i,
                                                  Relative_alpha_i,heading,pitch,speed,ready_missile,Availability,
                                                  lost_flag, jammed, be_seen_enemyleader, be_seen_myleader, be_seen))
                enemy_allplane_info.append(enemy_info)
        if len(self.enemy_plane)!=10:
            for i in range(10-len(self.enemy_plane)):
                Relative_X_i = np.array([0])
                Relative_Y_i = np.array([0])
                Relative_Z_i = np.array([0])
                Relative_dist_i = np.array([0])
                Relative_theta_i = np.array([0])
                Relative_alpha_i = np.array([0])
                heading = np.array([0])
                pitch = np.array([0])
                speed = np.array([0])
                ready_missile = np.array([0])
                Availability = np.array([0])
                lost_flag = np.array([0])
                jammed = np.array([0])
                be_seen_enemyleader = np.array([0])
                be_seen_myleader = np.array([0])
                be_seen = np.array([0])
                enemy_info = np.concatenate((Relative_X_i,Relative_Y_i,Relative_Z_i,Relative_dist_i,Relative_theta_i,
                                                Relative_alpha_i,heading,pitch,speed,ready_missile,Availability,
                                                lost_flag, jammed, be_seen_enemyleader, be_seen_myleader, be_seen))
                enemy_allplane_info.append(enemy_info)
        enemy_allplane_info = np.array(enemy_allplane_info)
        enemy_allplane_info = np.concatenate(enemy_allplane_info)
        return enemy_allplane_info
    
    # 我方每一个智能体的obs
    def get_my_agent_obs(self):
        my_plane_obs = []
        for my_plane in self.my_plane:
            my_plane_obs.append(self.get_current_control_info(my_plane))
        return np.array(my_plane_obs)
    
    # 获取全局obs
    def get_global_obs(self):
        global_my_score = np.array([self.my_score])
        global_enemy_score = np.array([self.enemy_score])
        global_sim_time = np.array([self.sim_time])
        global_my_missile = np.array([self.my_ready_weapon])
        global_ennmy_missile = np.array([self.enemy_left_weapon])
        # 归一化处理
        global_my_score[0] /= 68.8
        global_enemy_score[0] /= 68.8
        global_sim_time[0] = (20*60-global_sim_time[0])/20/60
        global_my_missile[0] /= 24
        global_ennmy_missile[0] /= 24
        global_info = np.concatenate((global_my_score,global_enemy_score,global_sim_time,global_my_missile,global_ennmy_missile))
        return global_info

    # 获得全局state
    def get_global_state(self):
        my_state = []
        for cur_plane in self.my_plane:
            Identification_my = np.array([0]) if cur_plane.Identification == "红方" else np.array([1])
            Type_my = np.array([0]) if cur_plane.Type == 1 else np.array([1])
            Availability_my = np.array([cur_plane.Availability])
            X_my = np.array([0])
            Y_my = np.array([0])
            Z_my = np.array([0])
            Heading_my = np.array([0])
            Pitch_my = np.array([0])
            Roll_my = np.array([0])
            Speed_my = np.array([0])
            close_missile_my = self.get_relative_missile_info(cur_plane, cur_plane.close_missile)
            center_time = np.array([0])
            LeftWeapon_my = np.array([0])
            do_jam = np.array([0])
            jammed = np.array([0])
            be_seen_enemyleader = np.array([0])
            be_seen_myleader = np.array([0])
            be_seen = np.array([0])
            threat_ratio = np.array([0])
            if Availability_my:
                X_my = np.array([cur_plane.X])
                Y_my = np.array([cur_plane.Y])
                Z_my = np.array([cur_plane.Z])
                Heading_my = np.array([cur_plane.Heading])
                Pitch_my = np.array([cur_plane.Pitch])
                Roll_my = np.array([cur_plane.Roll])
                Speed_my = np.array([cur_plane.Speed])
                center_time = np.array([cur_plane.center_time])
                LeftWeapon_my = np.array([cur_plane.ready_missile])
                do_jam = np.array([cur_plane.do_jam])
                jammed = np.array([self.sim_time - cur_plane.jammed])
                be_seen_enemyleader = np.array([cur_plane.be_seen_leader])
                be_seen_myleader = np.array([cur_plane.be_seen_myleader])
                be_seen = np.array([cur_plane.be_seen])
                threat_ratio = np.array([cur_plane.threat_ratio])
            current_control_info = np.concatenate((Identification_my, Type_my, Availability_my, X_my,
                                                Y_my, Z_my, Heading_my, Pitch_my, Roll_my, Speed_my,
                                                close_missile_my, center_time, do_jam, LeftWeapon_my,
                                                jammed, be_seen_enemyleader, be_seen_myleader, be_seen,
                                                threat_ratio))
            # 归一化处理
            current_control_info[3] /= 150000  # X_my:[-1, 1]
            current_control_info[4] /= 150000  # Y_my:[-1, 1]
            current_control_info[5] = (current_control_info[5] - 2000) / 13000 if cur_plane.Type == 1 else (current_control_info[5] - 2000) / 8000  # Z_my:[0, 1]
            current_control_info[6] /= 2*math.pi  # Heading_my:[-1, 1]
            current_control_info[7] /= (math.pi/2)  # Pitch_my:[-1, 1]
            current_control_info[8] /= math.pi  # Roll_my:[-1, 1]
            current_control_info[9] = (current_control_info[9] - 150) / 250 if cur_plane.Type == 1 else (current_control_info[9] - 100) / 200  # Speed:[0, 1]
            current_control_info[11] /= 10*60  # center_time:[0, 1]
            current_control_info[13] /= (4 if cur_plane.Type == 1 else 2)  # LeftWeapon_my:[0, 1]
            current_control_info[14] = 7/(current_control_info[14]+1)  # jammed:[0, 1]
            current_control_info[18] /= 1000  # threat_ratio:[0, 1]
            my_state.append(current_control_info)
        my_state = np.array(my_state)
        my_state = np.concatenate(my_state)
        enemy_allplane_info = []
        if len(self.enemy_plane):
            for cur_plane in self.enemy_plane:
                Identification_enemy = np.array([0]) if cur_plane.Identification == "红方" else np.array([1])
                Type_enemy = np.array([0]) if cur_plane.Type == 1 else np.array([1])
                Availability_enemy = np.array([cur_plane.Availability])
                X_enemy = np.array([0])
                Y_enemy = np.array([0])
                Z_enemy = np.array([0])
                Heading_enemy = np.array([0])
                Pitch_enemy = np.array([0])
                Roll_enemy = np.array([0])
                Speed_enemy = np.array([0])
                close_missile_enemy = self.get_relative_missile_info(cur_plane, cur_plane.close_missile)
                center_time = np.array([0])
                LeftWeapon_enemy = np.array([0])
                lost_flag = np.array([0])
                jammed = np.array([0])
                be_seen_enemyleader = np.array([0])
                be_seen_myleader = np.array([0])
                be_seen = np.array([0])
                if Availability_my:
                    X_enemy = np.array([cur_plane.X])
                    Y_enemy = np.array([cur_plane.Y])
                    Z_enemy = np.array([cur_plane.Z])
                    Heading_enemy = np.array([cur_plane.Heading])
                    Pitch_enemy = np.array([cur_plane.Pitch])
                    Roll_enemy = np.array([cur_plane.Roll])
                    Speed_enemy = np.array([cur_plane.Speed])
                    center_time = np.array([cur_plane.center_time])
                    LeftWeapon_enemy = np.array([cur_plane.ready_missile])
                    if cur_plane.lost_flag:
                        lost_flag = np.array([1])
                    jammed = np.array([self.sim_time - cur_plane.jammed])
                    be_seen_enemyleader = np.array([cur_plane.be_seen_leader])
                    be_seen_myleader = np.array([cur_plane.be_seen_myleader])
                    be_seen = np.array([cur_plane.be_seen])
                current_control_info = np.concatenate((Identification_enemy, Type_enemy, Availability_enemy, X_enemy,
                                                    Y_enemy, Z_enemy, Heading_enemy, Pitch_enemy, Roll_enemy, Speed_enemy,
                                                    close_missile_enemy, center_time, LeftWeapon_enemy, lost_flag,
                                                    jammed, be_seen_enemyleader, be_seen_myleader, be_seen))
                # 归一化处理
                current_control_info[3] /= 150000  # X_my:[-1, 1]
                current_control_info[4] /= 150000  # Y_my:[-1, 1]
                current_control_info[5] = (current_control_info[5] - 2000) / 13000 if cur_plane.Type == 1 else (current_control_info[5] - 2000) / 8000  # Z_my:[0, 1]
                current_control_info[6] /= 2*math.pi  # Heading_my:[-1, 1]
                current_control_info[7] /= (math.pi/2)  # Pitch_my:[-1, 1]
                current_control_info[8] /= math.pi  # Roll_my:[-1, 1]
                current_control_info[9] = (current_control_info[9] - 150) / 250 if cur_plane.Type == 1 else (current_control_info[9] - 100) / 200  # Speed:[0, 1]
                current_control_info[11] /= 10*60  # center_time:[0, 1]
                current_control_info[12] /= (4 if cur_plane.Type == 1 else 2)  # LeftWeapon_my:[0, 1]
                current_control_info[13] = 7/(current_control_info[13]+1)  # jammed:[0, 1]
                enemy_allplane_info.append(current_control_info)
            
        if len(self.enemy_plane)!=10:
            for i in range(10-len(self.enemy_plane)):
                Identification_enemy = np.array([0]) if self.side == -1 else np.array([1])
                Type_enemy = np.array([0])
                Availability_enemy = np.array([1])
                X_enemy = np.array([0])
                Y_enemy = np.array([0])
                Z_enemy = np.array([0])
                Heading_enemy = np.array([0])
                Pitch_enemy = np.array([0])
                Roll_enemy = np.array([0])
                Speed_enemy = np.array([0])
                close_missile_enemy = self.get_relative_missile_info(None, None)
                center_time = np.array([0])
                LeftWeapon_enemy = np.array([0])
                lost_flag = np.array([0])
                jammed = np.array([0])
                be_seen_enemyleader = np.array([0])
                be_seen_myleader = np.array([0])
                be_seen = np.array([0])
                enemy_info = np.concatenate((Identification_enemy, Type_enemy, Availability_enemy, X_enemy,
                                            Y_enemy, Z_enemy, Heading_enemy, Pitch_enemy, Roll_enemy, Speed_enemy,
                                            close_missile_enemy, center_time, LeftWeapon_enemy, lost_flag,
                                            jammed, be_seen_enemyleader, be_seen_myleader, be_seen))
                enemy_allplane_info.append(enemy_info)
        enemy_allplane_info = np.array(enemy_allplane_info)
        enemy_allplane_info = np.concatenate(enemy_allplane_info)
        state = np.concatenate((my_state, enemy_allplane_info))
        return state
        
    # 动作映射
    def action2cmd(self, actions):
        cmd_list = []
        actions_mask = self.actions_masked()
        for i,action in enumerate(actions):
            action_mask = actions_mask[i]
            # 对智能体的动作进行掩码处理
            action_mask = action_mask != 1
            action[action_mask] = -99999
            max_index = np.argmax(action)
            plane = self.get_body_info_by_id(self.my_plane, self.index_to_id[i])
            self.selected_module(max_index)(cmd_list, plane)
        return cmd_list

    # 动作掩码
    def actions_masked(self):
        actions_mask = []
        for i in range(10):
            action_space = self.num_azimuth*self.num_elevation+3
            action_mask = np.ones(action_space)
            plane = self.get_body_info_by_id(self.my_plane, self.index_to_id[i])
            if plane.Type==2:
                action_mask[self.num_azimuth*self.num_elevation+2] = 0
            elif self.sim_time - plane.last_jam<60:
                action_mask[self.num_azimuth*self.num_elevation+2] = 0
            else:
                have_enemy_jam = False
                for enemy in self.enemy_plane:
                    if plane.can_see(enemy,see_factor=0.9):
                        have_enemy_jam = True
                if have_enemy_jam==False:
                    action_mask[self.num_azimuth*self.num_elevation+2] = 0
            have_enemy_attack = False
            for enemy in self.enemy_plane:
                if enemy.lost_flag==0 and plane.can_attack(enemy) and plane.ready_missile>0:
                    have_enemy_attack = True
                    break
            if have_enemy_attack==False:
                action_mask[self.num_azimuth*self.num_elevation] = 0
            have_enemy_follow = False
            for enemy in self.enemy_plane:
                if enemy.lost_flag==0:
                    have_enemy_follow = True
            if have_enemy_follow==False:
                action_mask[self.num_azimuth*self.num_elevation+1] = 0
            actions_mask.append(action_mask)
        actions_mask = np.array(actions_mask)
        return actions_mask
    
    # 选择模块
    def selected_module(self, action):
        MODULE = {
            # 0: self.go_ahead,
            # 1: self.left_turn,
            # 2: self.right_turn,
            # 3: self.up_sky,
            # 4: self.down_ground,
            # 5: self.attack_enemy,
            # 6: self.follow_enemy,
            # 7: self.activate_jam
            self.num_azimuth*self.num_elevation: self.attack_enemy,
            self.num_azimuth*self.num_elevation+1: self.follow_enemy,
            self.num_azimuth*self.num_elevation+2: self.activate_jam
        }
        return MODULE[action]
    
    # 更新决策
    def update_rl_info(self, obs_side):
        self.update_entity_info(obs_side)
        my_obs = self.get_my_agent_obs()
        # global_obs = self.get_global_obs()
        global_obs = self.get_global_state()
        reward = np.array(self.rewards)
        done = np.array(self.done)
        info = {}
        actions_mask = self.actions_masked()
        return my_obs, global_obs, reward, done, actions_mask, info
    
    # # 根据动作映射至命令
    # def action2order(self, action, cmd_list):
    #     for i, act in enumerate(action):
    #         plane = self.get_body_info_by_id(self.my_plane, self.index_to_id[i])
    #         self.selected_module(act)(cmd_list, plane)

    # 根据动作映射至命令
    def action2order(self, action, cmd_list):
        for i, act in enumerate(action):
            plane = self.get_body_info_by_id(self.my_plane, self.index_to_id[i])
            if act<self.num_azimuth*self.num_elevation:
                self.move_action(act, cmd_list, plane)
            else:
                self.selected_module(act)(cmd_list, plane)
    
    def get_sector_centers(self, r=1000):
        """
        返回一个球面上以当前位置为球心，半径为r，以heading为方向，pitch为俯仰角，
        广角为90度的扇体，将其均分为num_azimuth*num_elevation个区域，每个区域
        的中心点坐标。

        参数：
        - r: 球半径

        返回值：
        - 一个包含所有区域中心点坐标的列表，每个元素为一个二元组 (azimuth, elevation)
        """
        # 计算扇形范围
        heading = 45
        pitch = 30
        sector_centers = []
        heading_pitch = [None for _ in range(self.num_azimuth*self.num_elevation)]
        azimuth_step = 2*heading/self.num_azimuth
        elevation_step = 2*pitch/self.num_elevation
        for i in range(self.num_azimuth):
            for j in range(self.num_elevation):
                # 计算当前区域的方位角和仰角
                azimuth = -heading + (i + 0.5) * azimuth_step
                elevation = -pitch + (j + 0.5) * elevation_step   
                # 将方位角和仰角转换为弧度
                azimuth_rad = math.radians(azimuth)
                elevation_rad = math.radians(elevation)
                heading_pitch[i*self.num_elevation+j] = (azimuth_rad, elevation_rad)
                # 计算当前区域的中心点坐标
                x = r * math.cos(elevation_rad) * math.cos(azimuth_rad)
                y = r * math.cos(elevation_rad) * math.sin(azimuth_rad)
                z = r * math.sin(elevation_rad)
                pos = np.array([x, y, z])
                # 将中心点坐标添加到列表中
                sector_centers.append(pos)

        return sector_centers, heading_pitch

    def get_sector_index(self, azimuth, elevation):
        """
        给定球半径r、当前位置的方向heading和俯仰角pitch，以及一个方位角和仰角，
        返回该方向所属的区域索引。

        参数：
        - r: 球半径
        - azimuth: 方位角度（0-360）
        - elevation: 仰角度（0-90）
        - num_azimuth: 方位角均分数，默认为9
        - num_elevation: 仰角均分数，默认为6

        返回值：
        - 该方向所属的区域索引，从0开始
        """
        r = 1000
        sector_centers,_ = self.get_sector_centers(r)
        # 将给定方向转换为球面坐标
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)
        x = r * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = r * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = r * math.sin(elevation_rad)

        # 查找最近的区域中心点
        min_dist = float('inf')
        min_index = -1
        for i, center in enumerate(sector_centers):
            dist = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        return min_index
    
    # 移动
    def move_action(self, act, cmd_list, plane):
        if self.heading_pitch is None:
            _, self.heading_pitch = self.get_sector_centers(r=1000)
        heading = self.heading_pitch[act][0]+plane.Heading
        pitch = self.heading_pitch[act][1]
        new_dir = TSVector3.calorientation(heading, pitch)
        new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, plane.Speed*20))
        route_list = [new_pos,]
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, plane.move_speed,
                                                    plane.para["move_max_acc"], plane.para["move_max_g"]))

    # 前进
    def go_ahead(self,cmd_list,plane):
        new_dir = TSVector3.calorientation(plane.Heading, 0)
        new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, plane.Speed*6))
        route_list = [new_pos,]
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, plane.move_speed,
                                                    plane.para["move_max_acc"], plane.para["move_max_g"]))
    # 左转
    def left_turn(self,cmd_list,plane):
        new_dir = TSVector3.calorientation(plane.Heading-math.pi/6, 0)
        new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, plane.Speed*6))
        route_list = [new_pos,]
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, plane.move_speed,
                                                    plane.para["move_max_acc"], plane.para["move_max_g"]))
    # 右转
    def right_turn(self,cmd_list,plane):
        new_dir = TSVector3.calorientation(plane.Heading+math.pi/6, 0)
        new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, plane.Speed*6))
        route_list = [new_pos,]
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, plane.move_speed,
                                                    plane.para["move_max_acc"], plane.para["move_max_g"]))
    # 上天
    def up_sky(self,cmd_list,plane):
        new_dir = TSVector3.calorientation(plane.Heading, math.pi/5.5)
        new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, plane.Speed*6))
        route_list = [new_pos,]
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, plane.move_speed,
                                                    plane.para["move_max_acc"], plane.para["move_max_g"]))
    # 入地
    def down_ground(self,cmd_list,plane):
        new_dir = TSVector3.calorientation(plane.Heading, -math.pi/5.5)
        new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, plane.Speed*6))
        route_list = [new_pos,]
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, plane.move_speed,
                                                    plane.para["move_max_acc"], plane.para["move_max_g"]))
    # 攻击
    def attack_enemy(self,cmd_list,plane):
        threat_plane_list = []
        for enemy in self.enemy_plane:
            if enemy.lost_flag==0 and plane.can_attack(enemy) and plane.ready_missile>0:
                threat_plane_list.append(enemy.ID)
        if len(threat_plane_list):
            threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).ready_missile, reverse=True)
            threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).wing_plane, reverse=False)
            threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).Type, reverse=False)
            threat_plane_list = sorted(threat_plane_list, key=lambda d: TSVector3.distance(self.get_body_info_by_id(self.enemy_plane,d).pos3d, plane.pos3d), reverse=False)
            factor_fight = 1
            cmd_list.append(env_cmd.make_attackparam(plane.ID, threat_plane_list[0], factor_fight))
    # 跟踪
    def follow_enemy(self,cmd_list,plane):
        threat_plane_list = []
        for enemy in self.enemy_plane:
            if enemy.lost_flag==0:
                threat_plane_list.append(enemy.ID)
        if len(threat_plane_list):
            threat_plane_list = sorted(threat_plane_list, key=lambda d: TSVector3.distance(self.get_body_info_by_id(self.enemy_plane,d).pos3d, plane.pos3d), reverse=False)
            if plane.ready_missile>0:
                threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).ready_missile, reverse=True)
                threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).wing_plane, reverse=False)
            threat_plane_list = sorted(threat_plane_list, key=lambda d: TSVector3.distance(self.get_body_info_by_id(self.enemy_plane,d).pos3d, plane.pos3d), reverse=False)
            enemy_plane = self.get_body_info_by_id(self.enemy_plane, threat_plane_list[0])
            if TSVector3.distance(plane.pos3d, enemy_plane.pos3d)<15000:
                if plane.move_speed > enemy_plane.Speed:
                    plane.move_speed = enemy_plane.Speed
            cmd_list.append(env_cmd.make_followparam(plane.ID, enemy_plane.ID, plane.move_speed, 
                                                     plane.para['move_max_acc'], plane.para['move_max_g']))
    # 干扰
    def activate_jam(self,cmd_list,plane):
        plane.do_jam = True
        plane.last_jam = self.sim_time+7
        for myplane in self.my_plane:
            if plane.ID != myplane.ID and plane.can_see(myplane, see_factor=0.9, jam_dis=117600):
                myplane.jammed = self.sim_time+1
        for enemy in self.enemy_plane:
            if plane.ID != enemy.ID and plane.can_see(enemy, see_factor=0.99, jam_dis=117600):
                enemy.jammed = self.sim_time+1
        cmd_list.append(env_cmd.make_jamparam(plane.ID))
    
    # 计算是否逃不掉了
    def death_to_death(self, plane,  missile, new_plane_pos=None, radius = 180):
        dis, t = self.shortest_distance_between_linesegment(plane, missile, new_plane_pos)
        if dis<radius:
            return True
        else:
            return False

    # 临死发射全部导弹攻击敌方:都得死,玉石俱焚
    def all_death(self, plane, cmd_list):   
        threat_plane_list = []
        # print(self.sim_time,plane.Name, "玉石俱焚")
        for enemy in self.enemy_plane:
            if enemy.lost_flag==0 and plane.can_attack(enemy, attack_dis=self.attack_distance) and TSVector3.distance(plane.pos3d, enemy.pos3d)/1000+self.sim_time+2<20*60:
                threat_plane_list.append(enemy.ID)
        if len(threat_plane_list):
            threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).ready_missile, reverse=True)
            threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).wing_plane, reverse=False)
            threat_plane_list = sorted(threat_plane_list, key=lambda d: self.get_body_info_by_id(self.enemy_plane,d).Type, reverse=False)
            threat_plane_list = sorted(threat_plane_list, key=lambda d: TSVector3.distance(self.get_body_info_by_id(self.enemy_plane,d).pos3d, plane.pos3d), reverse=False)
        for threat_plane_id in threat_plane_list:
            threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_plane_id)
            if plane.ready_missile>0 and plane.ID not in threat_plane.ready_attacked:
                attack_time = 1
                if TSVector3.distance(threat_plane.pos3d, plane.pos3d)<self.attack_distance and plane.ready_missile>1:
                    attack_time = 2
                while(attack_time>0):
                    threat_ID = threat_plane_id
                    attack_time -= 1
                    if plane.ID not in threat_plane.ready_attacked:
                        threat_plane.ready_attacked.append(plane.ID)
                    if threat_plane.ID not in plane.ready_attack:
                        plane.ready_attack.append(threat_plane.ID)
                    factor_fight = 1
                    cmd_list.append(env_cmd.make_attackparam(plane.ID, threat_ID, factor_fight))
                    plane.do_be_fired = True
                    plane.ready_missile -= 1
                    self.my_score -= self.missile_score
        
    #初始化我方飞机位置
    def init_pos(self, cmd_list):
        # 初始化部署
        if self.sim_time == 2:
            leader_plane_1 = self.my_leader_plane[0]
            leader_plane_2 = self.my_leader_plane[1]
            # 初始化有人机位置
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -125000 * self.side, 60000, 9000, 400, self.init_direction))
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_2.ID, -125000 * self.side, -60000, 9000, 400, self.init_direction))
            for i, plane in enumerate(self.my_uav_plane):
                if i < 6:
                    if i != 5:
                        cmd_list.append(
                            env_cmd.make_entityinitinfo(plane.ID, -125000 * self.side, 150000 - (i+1) * 50000, 9000, 300, self.init_direction))
                    else:
                        cmd_list.append(
                            env_cmd.make_entityinitinfo(plane.ID, -125000 * self.side, 0, 9000, 300, self.init_direction))
                else:
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(plane.ID, -145000 * self.side, 75000 - (i+1)%3 * 50000, 9000, 300, self.init_direction))


    def init_moving(self, cmd_list):
        init_direction = self.init_direction
        distance = 10*300
        new_dir = TSVector3.calorientation(init_direction/180*math.pi, 0)
        leader_plane_1 = self.my_leader_plane[0]
        leader_plane_2 = self.my_leader_plane[1]
        new_pos = TSVector3.plus(leader_plane_1.pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 9000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(leader_plane_1.ID, route_list, 200, 1, 6))

        new_pos = TSVector3.plus(leader_plane_2.pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 9000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(leader_plane_2.ID, route_list, 200, 1, 6))

        new_pos = TSVector3.plus(self.my_uav_plane[0].pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 9000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(self.my_uav_plane[0].ID, route_list, 300, 2, 6))

        new_pos = TSVector3.plus(self.my_uav_plane[1].pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 9000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(self.my_uav_plane[1].ID, route_list, 300, 2, 6))

        new_pos = TSVector3.plus(self.my_uav_plane[2].pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 6000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(self.my_uav_plane[2].ID, route_list, 295, 2, 6))

        new_pos = TSVector3.plus(self.my_uav_plane[3].pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 6000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(self.my_uav_plane[3].ID, route_list, 295, 2, 6))

        new_pos = TSVector3.plus(self.my_uav_plane[4].pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 3000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(self.my_uav_plane[4].ID, route_list, 280, 2, 6))

        new_pos = TSVector3.plus(self.my_uav_plane[5].pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 3000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(self.my_uav_plane[5].ID, route_list, 285, 2, 6))

        new_pos = TSVector3.plus(self.my_uav_plane[6].pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 5000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(self.my_uav_plane[6].ID, route_list, 300, 2, 6))

        new_pos = TSVector3.plus(self.my_uav_plane[7].pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 5000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(self.my_uav_plane[7].ID, route_list, 300, 2, 6))
        
    # 我方飞机无威胁且空闲情况下飞机路径
    def init_move(self, cmd_list):
        if self.sim_time<5*60:
            self.init_moving(cmd_list)
            return
        for plane_ID in self.free_plane:
            plane = self.get_body_info_by_id(self.my_plane, plane_ID)
            if plane.Type==2 and len(self.enemy_plane):
                if self.sim_time%2:
                    relative_pitch = -math.pi/6
                else:
                    relative_pitch = math.pi/6
            else:
                relative_pitch = 0
                enemy_num = 0
                for enemy in self.enemy_plane:
                    if enemy.Availability and enemy.ready_missile>0:
                        if TSVector3.distance(enemy.pos3d, plane.pos3d)<80000:
                            enemy_num += 15000/(TSVector3.distance(enemy.pos3d, plane.pos3d)+2000)
                            relative_pitch -= enemy.Pitch*15000/(TSVector3.distance(enemy.pos3d, plane.pos3d)+2000)
                if enemy_num>0:
                    relative_pitch /= enemy_num
                
            if (self.side == 1 and plane.X > self.bound * self.side) or (self.side == -1 and plane.X < self.bound * self.side):
                init_direction = (self.init_direction + 180) % 360
            else:
                if (plane.Heading*180/math.pi - self.init_direction)<5 or self.sim_time<600:
                    init_direction = self.init_direction
                else:
                    init_direction = (self.init_direction + 180) % 360
            distance = 15*plane.para["move_max_speed"]
            new_dir = TSVector3.calorientation(init_direction/180*math.pi, relative_pitch)
            new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, distance))
            route_list = [new_pos,]
            if init_direction != self.init_direction or len(self.enemy_leader_plane) == 2:
                center_point = {"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000}
                center_point = TSVector3.plus(plane.pos3d, TSVector3.multscalar(TSVector3.normalize(TSVector3.minus(center_point,plane.pos3d)), plane.Speed*6))
                center_point['Z'] = new_pos['Z']
                route_list = [center_point,]
            if len(self.enemy_plane) and self.is_in_center(plane, center_radius=40000)==False:
                route_list = [{"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000},]
            # 没有敌人，朝对方中心点移动
            if plane.Type==1:
                if plane.move_order==None:
                    plane.move_order="巡航0"
                    plane.total_threat_flag = None
                    cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 300, 1, 6))     
                else:
                    print(plane.ID, plane.move_order, "巡航0")
            elif plane.Type==2:
                target_pos = {"X": 135000 * self.side, "Y": (random.random()*2-1)*3000, "Z": 2000}
                new_heading = TSVector3.calheading(TSVector3.minus(target_pos,plane.pos3d))
                new_dir = TSVector3.calorientation(new_heading,0)
                new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, plane.Speed*6))
                if (self.side==1 and plane.X>5000) or (self.side==-1 and plane.X<-5000):
                    pass
                else:
                    new_pos['Z'] = 3000

                route_list = [new_pos,]
                if plane.move_order==None:
                    plane.move_order="巡航"
                    plane.total_threat_flag = None
                    cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 300, 2, 6))
                else:
                    print(plane.ID,plane.move_order,"巡航")

    # 找到可以攻击敌方飞机的己方飞机
    def can_attack_plane(self, threat_plane_list):
        can_attack_dict = {}
        for my_plane in self.my_plane:# 有进入必杀的敌机，直接必杀
            if my_plane.Availability==0 or my_plane.ready_missile==0:
                continue
            dis = 7200
            enemy_plane = None
            for threat_plane_id in threat_plane_list:
                threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_plane_id)
                if dis>TSVector3.distance(threat_plane.pos3d, my_plane.pos3d) and my_plane.can_see(threat_plane,see_factor=0.99) and threat_plane.ID not in my_plane.ready_attack:
                    dis = TSVector3.distance(threat_plane.pos3d, my_plane.pos3d)
                    enemy_plane = threat_plane
            if  enemy_plane:
                distance = TSVector3.distance(my_plane.pos3d,enemy_plane.pos3d)
                if distance<4900:
                    can_attack_dict[my_plane.ID] = enemy_plane.ID
                elif distance<7200:
                    enemy_can_see_me = False
                    for enemy in self.enemy_plane:
                        if enemy.can_see(my_plane,see_factor=1):
                            enemy_can_see_me = True
                    if enemy_can_see_me==False or distance<5000:
                        can_attack_dict[my_plane.ID] = enemy_plane.ID
        
        for my_plane in self.my_plane:
            if my_plane.Availability==0 or my_plane.ready_missile==0 or can_attack_dict.get(my_plane.ID) is not None:
                continue
            dis = 99999999
            enemy_plane = None
            for threat_plane_id in threat_plane_list:
                threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_plane_id)
                attack_dis = self.attack_distance
                if threat_plane.Type==2:
                    attack_dis += 100
                in_range = my_plane.can_attack(threat_plane, attack_dis)
                if dis>TSVector3.distance(threat_plane.pos3d, my_plane.pos3d) and in_range and threat_plane.ID not in my_plane.ready_attack:
                    dis = TSVector3.distance(threat_plane.pos3d, my_plane.pos3d)
                    enemy_plane = threat_plane
            if enemy_plane:
                attack_plane = my_plane
                enemy_can_see_me = False
                for enemy in self.enemy_plane:
                    if enemy.can_see(attack_plane,see_factor=1):
                        enemy_can_see_me = True
                if  self.revenge or (self.my_score>self.enemy_score+self.win_score and enemy_can_see_me == False) or enemy_plane.be_seen_leader:
                    can_attack_dict[attack_plane.ID] = enemy_plane.ID

        return can_attack_dict

    # 飞机拐弯时间
    def get_turn_time(self,leader, enemy, see_factor=1):
        my_turn_theta = abs(
            leader.pi_bound(leader.XY2theta(enemy.X - leader.X, enemy.Y - leader.Y) - leader.Heading))
        look_theta = math.pi*leader.para['radar_heading']/180
        if my_turn_theta > look_theta*see_factor:
            my_turn_theta -= look_theta*see_factor
        else:
            my_turn_theta = 0
        my_turn_t = my_turn_theta / (leader.para['move_max_g'] * 9.8) * leader.Speed
        return my_turn_t

    # 跟踪可以进行干扰但不在探测范围的敌机
    def find_way_jam(self, jam_plane, middle_enemy_id, cmd_list):
        if middle_enemy_id==None:
            return
        jam_enemy = self.get_body_info_by_id(self.enemy_plane, middle_enemy_id)
        if jam_enemy.lost_flag==0:
            if jam_plane.move_order==None:
                jam_plane.do_turn = True
                jam_plane.total_threat_flag = None
                jam_plane.move_order="搜寻干扰1"
                cmd_list.append(env_cmd.make_followparam(jam_plane.ID, jam_enemy.ID, jam_plane.move_speed, jam_plane.para['move_max_acc'], jam_plane.para['move_max_g']))
            else:
                print(jam_plane.ID,jam_plane.move_order,"搜寻干扰1")
        else:
            if jam_plane.move_order==None:
                jam_plane.do_turn = True
                jam_plane.total_threat_flag = None
                jam_plane.move_order="搜寻干扰2"
                cmd_list.append(env_cmd.make_linepatrolparam(jam_plane.ID, [jam_enemy.pos3d,], jam_plane.move_speed,
                                                            jam_plane.para["move_max_acc"], jam_plane.para["move_max_g"]))
            else:
                print(jam_plane.ID,jam_plane.move_order,"搜寻干扰2")

    # 判断是否实时胜利
    def win_now(self):
        self.enemy_score_before = self.enemy_score
        self.my_score = 0
        self.enemy_score = 0
        self.my_center_time = 0
        self.enemy_center_time = 0
        for plane in self.my_plane:
            if plane.Availability == 1:
                if plane.Type==1:
                    self.my_center_time += plane.center_time
                self.my_score += plane.ready_missile*self.missile_score
                if plane.Type == 2:
                    self.my_score += self.uav_score
                else:
                    self.my_score += self.leader_score
        for plane in self.enemy_plane:
            if plane.Availability == 1:
                if plane.Type==1:
                    self.enemy_center_time += plane.center_time
                self.enemy_score += plane.ready_missile*self.missile_score
                if plane.Type == 2:
                    self.enemy_score += self.uav_score
                else:
                    self.enemy_score += self.leader_score
        if len(self.enemy_plane) != 10:
            self.enemy_score += (2 - len(self.enemy_leader_plane))*(self.leader_score + self.missile_score * 4)
            self.enemy_score += (8 - len(self.enemy_uav_plane))*(self.uav_score + self.missile_score * 2)
            self.enemy_center_time = self.my_center_time
        # if self.sim_time>19*60+57:
        #     print(self.my_score ,self.enemy_score, self.my_center_time, self.enemy_center_time)
        if self.enemy_score<self.my_score:
            return True
        elif self.enemy_score==self.my_score and self.my_center_time>self.enemy_center_time:
            return True
        else:
            return False

    # 判断是否飞机实体在中心
    def is_in_center(self, plane, center_radius=50000):
        distance_to_center = (plane.X**2 + plane.Y**2 + (plane.Z - 9000)**2)**0.5
        if distance_to_center <= center_radius and plane.Z >= 2000 and plane.Z <= plane.para['area_max_alt']:
            return True
        return False
    
    # 判断智能体是否在合法范围内
    def is_in_legal_area(self, plane):
        total_dir = TSVector3.calorientation(plane.Heading, plane.Pitch)
        distance = plane.Speed+plane.plane_acc*0.5
        next_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(total_dir, distance))
        if abs(next_pos['X']) > 150000 or abs(next_pos['Y']) > 150000 or next_pos['Z']<2000 or next_pos['Z']>plane.para['area_max_alt']:
            return False
        elif abs(plane.X)>150000 or abs(plane.Y)>150000 or plane.Z>plane.para['area_max_alt'] or plane.Z<2000:
            return False
        else:
            return True

    # 合成威胁向量
    def synthetic_threat_vector(self, leader, see_dis=25000, seen=True):
        total_dir = {"X": 0, "Y": 0, "Z": 0}
        min_dis = 99999999
        threat_weight = 0
        threat_pitch = 0
        if self.enemy_left_weapon==0:
            return total_dir, min_dis, threat_pitch
        for enemy in self.enemy_plane:
            if enemy.Availability and enemy.ready_missile>0:
                dis = TSVector3.distance(enemy.pos3d, leader.pos3d)
                if seen:
                    if enemy.can_see(leader) and dis<see_dis:
                        if TSVector3.distance(enemy.pos3d, leader.pos3d) < min_dis:
                            min_dis = TSVector3.distance(enemy.pos3d, leader.pos3d)
                            threat_pitch = enemy.Pitch
                        tmp_dir2 = TSVector3.multscalar(TSVector3.minus(leader.pos3d, enemy.pos3d), math.pow(leader.para["safe_range"]/(dis+200),2))
                        threat_weight += math.pow(leader.para["safe_range"]/(dis+200),2)
                        total_dir = TSVector3.plus(tmp_dir2,total_dir)
                else:
                    if dis<see_dis:
                        if TSVector3.distance(enemy.pos3d, leader.pos3d) < min_dis:
                            min_dis = TSVector3.distance(enemy.pos3d, leader.pos3d)
                            threat_pitch = enemy.Pitch
                        if enemy.Type==1:
                            ratio = 1.2
                        else:
                            ratio = 1
                        tmp_dir2 = TSVector3.multscalar(TSVector3.minus(leader.pos3d, enemy.pos3d), ratio*math.pow(leader.para["safe_range"]/(dis+200),2))
                        threat_weight += ratio*math.pow(leader.para["safe_range"]/(dis+200),2)
                        total_dir = TSVector3.plus(tmp_dir2,total_dir)
        if len(self.enemy_plane)>5 and leader.ready_missile:
            need_same_heading = False
            if (total_dir != {"X": 0, "Y": 0, "Z": 0} or leader.Type==1) and min_dis>16000:
                if total_dir != {"X": 0, "Y": 0, "Z": 0}:
                    need_same_heading = True
                for plane in self.my_plane:
                    if plane.Availability and plane.ID != leader.ID:
                        dis = TSVector3.distance(plane.pos3d, leader.pos3d)
                        if dis<60000:
                            if need_same_heading:
                                target_heading = TSVector3.calheading(total_dir)
                                team_heading = TSVector3.calheading(TSVector3.minus(plane.pos3d, leader.pos3d))
                                if abs(leader.pi_bound(team_heading-target_heading))>math.pi/4:
                                    continue
                            if plane.ready_missile:
                                ratio = 0.5
                            else:
                                ratio = 1
                            tmp_dir2 = TSVector3.multscalar(TSVector3.minus(plane.pos3d, leader.pos3d), ratio*math.pow((leader.para["safe_range"]-1000)/(dis+200),2))
                            threat_weight += ratio*math.pow((leader.para["safe_range"]-1000)/(dis+200),2)
                            total_dir = TSVector3.plus(tmp_dir2,total_dir)
        elif len(self.enemy_plane)>5 and leader.ready_missile==0 and total_dir == {"X": 0, "Y": 0, "Z": 0}:
            for plane in self.my_plane:
                if plane.Availability and plane.ID != leader.ID and plane.be_in_danger and plane.follow_plane:
                    dis = TSVector3.distance(plane.pos3d, leader.pos3d)
                    godis = 60000
                    if leader.Type==1:
                        godis = 110000
                    if dis<godis:
                        tmp_dir2 = TSVector3.multscalar(TSVector3.minus(plane.pos3d, leader.pos3d), math.pow((leader.para["safe_range"]-1000)/(dis+200),2))
                        threat_weight += math.pow((leader.para["safe_range"]-1000)/(dis+200),2)
                        total_dir = TSVector3.plus(tmp_dir2,total_dir)
        if threat_weight>0:
            total_dir = TSVector3.multscalar(total_dir, 1/threat_weight)
            if min_dis>leader.para["safe_range"]:
                threat_pitch = TSVector3.calpitch(total_dir)
        return total_dir, min_dis, threat_pitch

    # 计算飞机与导弹相遇最小距离
    def shortest_distance_between_linesegment(self, plane, missile, next_plane_pos=None):
        if next_plane_pos!=None:
            heading = TSVector3.calheading(TSVector3.minus(next_plane_pos, plane.pos3d))
            pitch = TSVector3.calpitch(TSVector3.minus(next_plane_pos, plane.pos3d))
            plane_v = TSVector3.multscalar(TSVector3.calorientation(heading, pitch), plane.Speed+plane.plane_acc)
        else:
            plane_v = TSVector3.multscalar(TSVector3.calorientation(plane.Heading, plane.Pitch), plane.Speed+plane.plane_acc)
        missile_v = TSVector3.multscalar(TSVector3.calorientation(missile.Heading, missile.Pitch), missile.Speed+missile.missile_acc)
        plane_pos = plane.pos3d
        missile_pos = missile.pos3d
        a = (plane_v['X']-missile_v['X'])**2+(plane_v['Y']-missile_v['Y'])**2+(plane_v['Z']-missile_v['Z'])**2
        b = 2*((plane_pos['X']-missile_pos['X'])*(plane_v['X']-missile_v['X'])+(plane_pos['Y']-missile_pos['Y'])*(plane_v['Y']-missile_v['Y'])+(plane_pos['Z']-missile_pos['Z'])*(plane_v['Z']-missile_v['Z']))
        c = (plane_pos['X']-missile_pos['X'])**2 + (plane_pos['Y']-missile_pos['Y'])**2 + (plane_pos['Z']-missile_pos['Z'])**2
        ans_dis = math.sqrt((4*a*c-b*b)/4/a)
        ans_t = -b/2/a
        return ans_dis, ans_t+self.sim_time