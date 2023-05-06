import math
import random
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.agent import Agent
from agent.tink_AI_studio_nana.agent_base import Plane, Missile

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
        self.attack_distance = 10000
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
        # 间谍编队
        self.spy_plane = []
        # 当前时间
        self.sim_time = 0
        # 当前敌有人机还剩多少
        self.live_enemy_leader = 0
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
        # 我方空闲飞机
        self.free_plane = []

    def reset(self):
        self._init()

    def step(self,sim_time, obs_side, **kwargs):
        cmd_list = []
        self.sim_time = sim_time
        self.update_decision(obs_side, cmd_list)
        return cmd_list

    # 更新当前各个实体的存亡以及部分额外记录信息
    def update_dead_info(self):
        for plane in self.my_plane:
            plane.move_order = None
            plane.be_seen_leader = False
            plane.ready_attack = []
            if plane.follow_plane:
                enemy = self.get_body_info_by_id(self.enemy_plane, plane.follow_plane)
                if plane.ID not in enemy.followed_plane or plane.ready_missile==0:
                    plane.follow_plane = None
            if plane.Type==1 and self.sim_time-plane.last_jam>0:
                plane.do_jam = False
                plane.do_turn = False
            if plane.lost_flag and plane.Availability:
                plane.Availability = 0
                if plane.follow_plane is not None:
                    enemy = self.get_body_info_by_id(self.enemy_plane, plane.follow_plane)
                    if plane.ID in enemy.followed_plane:
                        enemy.followed_plane.remove(plane.ID)
                    plane.follow_plane = None
            elif plane.Availability and self.is_in_center(plane):
                plane.center_time += 1
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                    if plane.do_be_fired:
                        missile.be_fired = True
                if missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list and missile.lost_flag==0 and missile.loss_target==False:
                    plane.locked_missile_list.append(missile.ID)
                if missile.EngageTargetID == plane.ID and missile.loss_target==False:
                    short_dis, missile.arrive_time = self.shortest_distance_between_linesegment(plane, missile)
            plane.do_be_fired = False
            # 更新导弹追踪信息
            for missile_id in plane.locked_missile_list.copy():
                tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                be_seen = False
                for m_p in self.my_plane:
                    if m_p.can_see(tmp_missile, see_factor=0.97):
                        be_seen = True
                if (tmp_missile.lost_flag and (self.sim_time - tmp_missile.arrive_time > 1 or (be_seen and self.sim_time - tmp_missile.arrive_time>-1))) or tmp_missile.loss_target:
                    plane.locked_missile_list.remove(tmp_missile.ID)
            dis = 99999999
            plane.close_missile = None
            for missile_id in plane.locked_missile_list:
                missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if TSVector3.distance(plane.pos3d, missile.pos3d) < dis and missile.loss_target==False:
                    dis = TSVector3.distance(plane.pos3d, missile.pos3d)
                    plane.close_missile = missile    
        # 更新是否反击
        self.win_now()
        # if self.sim_time>19*60+57 and self.my_score>self.enemy_score:
        #     print("预测赢了:",self.my_score,'   ',self.enemy_score)
        change_dis = False
        self.live_enemy_leader = 0
        for plane in self.enemy_plane:
            plane.ready_attacked = []
            plane.be_seen_leader = False
            plane.be_seen = False
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
                            if tmp_missile.be_fired==False or tmp_missile.init_dis<self.attack_distance+1000:
                                change_dis = True
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
                        if tmp_missile.lost_flag and self.death_to_death(plane,tmp_missile,next_plane_pos,radius=100):
                            dead_flag += 1
                        if tmp_missile.loss_target==True:
                            dead_flag = -1
                    if dead_flag==-1:
                        plane.Availability = 0
                if self.is_in_legal_area(plane)==False:
                    plane.Availability = 0
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
            plane.ready_missile = plane.AllWeapon - len(plane.used_missile_list)
        # 计算对方是否存在僚机掩护
        for plane in self.enemy_plane:
            plane.wing_plane = 0
            if plane.Availability:
                for wing_plane in self.enemy_plane:
                    if wing_plane.Availability and plane.ID != wing_plane.ID and wing_plane.can_see(plane,see_factor=1):
                        plane.wing_plane += 1
        for spy_plane in self.spy_plane:             
            spy_enemy = None
            dis = 999999999
            spy_plane = self.get_body_info_by_id(self.my_plane, spy_plane)
            if len(spy_plane.locked_missile_list):
                spy_plane.is_spy = False
            if spy_plane!=None and spy_plane.ready_missile==0:
                spy_plane.is_spy = False
            elif spy_plane != None:
                for plane in self.enemy_plane:
                    if plane.Type==1 and plane.Availability and TSVector3.distance(spy_plane.pos3d,plane.pos3d)<dis:
                        dis = TSVector3.distance(spy_plane.pos3d,plane.pos3d)
                        spy_enemy = plane
            if spy_enemy != None:
                rel_heading = TSVector3.calheading(TSVector3.minus(spy_plane.pos3d,spy_enemy.pos3d))
                is_back = abs(spy_plane.pi_bound(rel_heading-spy_enemy.Heading))>90/180*math.pi
                if spy_enemy.can_see(spy_plane,see_factor=1)==False and is_back:
                    spy_plane.is_spy = False
        
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
                plane.move_speed = (plane.para["move_min_speed"]+plane.para["move_max_speed"])/2
                plane.threat_ratio = 0
                if len(plane.locked_missile_list):
                    plane.move_speed = plane.para["move_max_speed"]
                    plane.threat_ratio = 1000000
                else:
                    for enemy in self.enemy_plane:
                        if enemy.Availability and enemy.can_see(plane,see_factor=1.0) and enemy.ready_missile>0:
                        # if enemy.Availability and enemy.ready_missile>0:
                            enemy_dis = TSVector3.distance(plane.pos3d, enemy.pos3d)
                            speed_factor = math.tanh(math.pow(plane.para['safe_range']/(enemy_dis+2000),2))
                            plane.threat_ratio = max(plane.threat_ratio, math.pow(plane.para['safe_range']/(enemy_dis+2000),2))
                            tmp_move_speed = speed_factor*(plane.para["move_max_speed"]-plane.para["move_min_speed"]) + plane.para["move_min_speed"]
                            plane.move_speed = min(max(plane.move_speed, tmp_move_speed),plane.para["move_max_speed"])

        # 己方有人机信息
        self.my_leader_plane = self.get_body_info_by_type(self.my_plane, 1)
        # 己方无人机信息
        self.my_uav_plane = self.get_body_info_by_type(self.my_plane, 2)
        # 敌方有人机信息
        self.enemy_leader_plane = self.get_body_info_by_type(self.enemy_plane, 1)
        # 敌方无人机信息
        self.enemy_uav_plane = self.get_body_info_by_type(self.enemy_plane, 2)

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
    def update_decision(self, obs_side, cmd_list):
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
                        
             # 无人机生存之道模块      
            # 无人机脱离近距离攻击范围
            for plane in self.my_plane:
                if plane.ID in self.free_plane and plane.Type==2:
                    if plane.ready_missile:
                        total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(plane, see_dis=16000,seen=True)
                    else:
                        total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(plane, see_dis=26000,seen=False)
                    if total_dir != {"X": 0, "Y": 0, "Z": 0}:
                        if plane.wing_who!=None:
                            be_wing_plane = self.get_body_info_by_id(self.my_plane, plane.wing_who)
                            be_wing_plane.wing_plane = None
                            plane.wing_who = None
                        if self.is_in_center(plane,140000) == False:
                            tmp_dir2 = {"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000}
                            tmp_dir2 = TSVector3.minus(tmp_dir2, plane.pos3d)
                            total_dir = tmp_dir2
                            min_dis = 99999999
                        self.free_plane.remove(plane.ID)
                        if plane.ID in no_missile_plane:
                            no_missile_plane.remove(plane.ID)
                        total_heading = TSVector3.calheading(total_dir)
                        total_pitch = -1*threat_pitch
                        if min_dis>14000:
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
            # 干扰模块
            self.activate_jam(cmd_list)
            
            # 导弹隐身攻击模块
            jam_for_attack = []
            for leader in self.my_leader_plane:
                # 如果在跑路时也能提供干扰支持
                if leader.Availability and self.sim_time-leader.last_jam>60 and leader.do_jam==False:
                    total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(leader, see_dis=30000, seen=False)
                    if total_dir == {"X": 0, "Y": 0, "Z": 0}:
                        if len(leader.locked_missile_list)==0:
                            jam_for_attack.append(leader)
                        elif leader.close_missile and TSVector3.distance(leader.close_missile.pos3d, leader.pos3d)>19000:
                            jam_for_attack.append(leader)
            if len(jam_for_attack)>0: 
                my_missile = self.get_body_info_by_identification(self.missile_list, self.my_plane[0].Identification)
                for jam_plane in jam_for_attack:
                    can_jammed_enemy = {}
                    jammed_missile_list = []
                    for enemy in self.enemy_plane:
                        if enemy.lost_flag==0 and jam_plane.can_see(enemy,see_factor=1):
                            enemy.be_seen_leader = True
                            enemy.be_seen = True
                    for missile_id in my_missile.copy():
                        missile = self.get_body_info_by_id(self.missile_list, missile_id)
                        target_plane = self.get_body_info_by_id(self.enemy_plane, missile.EngageTargetID)
                        jam_turn_time = self.get_turn_time(jam_plane, target_plane)
                        # can_jam = jam_turn_time*missile.Speed<TSVector3.distance(missile.pos3d, target_plane.pos3d)<(jam_turn_time+6)*missile.Speed
                        can_jam = jam_turn_time*missile.Speed<TSVector3.distance(missile.pos3d, target_plane.pos3d)
                        can_see = TSVector3.distance(jam_plane.pos3d, target_plane.pos3d)<120000-jam_turn_time*target_plane.Speed
                        if target_plane.ID not in can_jammed_enemy and missile.lost_flag==0 and target_plane.lost_flag==0 and can_jam and missile.loss_target==False and can_see:
                            can_jammed_enemy[target_plane.ID] = jam_plane.pi_bound(TSVector3.calheading(TSVector3.minus(target_plane.pos3d,jam_plane.pos3d))-jam_plane.Heading)
                            my_missile.remove(missile_id)
                            jammed_missile_list.append(missile_id)
                    can_jammed_enemy = [key for key, value in sorted(can_jammed_enemy.items(), key=lambda d: d[1])]
                    if len(can_jammed_enemy):
                        if jam_plane.middle_enemy_plane==None:
                            jam_plane.middle_enemy_plane = can_jammed_enemy[int(len(can_jammed_enemy)/2)]
                        # 万一有人机遇到导弹威胁时middle有
                        middle_enemy = self.get_body_info_by_id(self.enemy_plane,jam_plane.middle_enemy_plane)
                        if jam_plane.can_see(middle_enemy,see_factor=0.6,jam_dis=117600):
                            jam_plane.do_jam = True
                            jam_plane.do_turn = False
                            cmd_list.append(env_cmd.make_jamparam(jam_plane.ID))
                            jam_plane.jammed = self.sim_time+1
                            jam_plane.middle_enemy_plane = None
                            jam_plane.last_jam = self.sim_time+7
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

            # 是否存在掉头攻击的可能
            for plane in self.my_plane:
                if plane.Availability and plane.ready_missile and plane.move_order==None and plane.do_turn==False and plane.do_jam==False and plane.ID in self.free_plane:
                    enemy_have_missile = False
                    for enemy in self.enemy_plane:
                        if enemy.Availability==0 and enemy.can_see(plane, jam_dis=plane.para["safe_range"]+1500) and enemy.ready_missile:
                            enemy_have_missile = True
                    if enemy_have_missile==False:
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
                # if leader.ID in self.free_plane and (self.my_left_weapon >2 or len(leader.used_missile_list)==4):
                if leader.ready_missile:
                    total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(leader, seen=False)
                else:
                    total_dir, min_dis, threat_pitch = self.synthetic_threat_vector(leader, see_dis=26000, seen=False)
                if self.enemy_left_weapon==0:
                    break
                if leader.ID in self.free_plane and (len(leader.used_missile_list)==4 or min_dis<leader.para["safe_range"]+5000):
                    if total_dir != {"X": 0, "Y": 0, "Z": 0}:
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
                            min_dis = 99999999
                        self.free_plane.remove(leader.ID)
                        total_heading = TSVector3.calheading(total_dir)
                        total_pitch = -1*threat_pitch
                        if min_dis>self.attack_distance:
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
                        elif min_dis<self.attack_distance:
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
                                if plane.follow_plane!=None and plane.ready_missile and plane.ID not in support_uav_plane_list and TSVector3.distance(plane.pos3d,leader.pos3d)<leader.para['radar_range']*0.9:
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

             # 飞机波浪式S型转圈视野扩展
            s_my_plane = sorted(self.my_plane, key=lambda d: d.ready_missile, reverse=False)
            if len(self.enemy_plane)>0:
                for plane in s_my_plane:
                    if plane.ID in self.free_plane:
                        plane.enemy_heading_range = [2*math.pi, 0]
                        plane.enemy_pitch_range = [math.pi/4, -math.pi/4]
                        can_see_enemy = False
                        for enemy in self.enemy_plane:
                            if TSVector3.distance(enemy.pos3d, plane.pos3d)<plane.para['radar_range'] and enemy.Availability and enemy.be_seen==False:
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
                                enemy.be_seen = True
                        if can_see_enemy==False:
                            for enemy in self.my_plane:
                                if TSVector3.distance(enemy.pos3d, plane.pos3d)<plane.para['radar_range'] and enemy.Availability and enemy.ID!=plane.ID:
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

            # 扩大视野模块
            # 在跟踪其他飞机
            for my_plane in self.first_formation:
                if my_plane.ID in self.free_plane and my_plane.ready_missile:
                    if my_plane.follow_plane:
                        enemy = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                        if my_plane.ID not in enemy.followed_plane:
                            my_plane.follow_plane = None
                    if my_plane.follow_plane == None and my_plane.is_spy==False:
                        dis = 999999999
                        for enemy_plane in self.enemy_plane:
                            if enemy_plane.lost_flag==0 and len(enemy_plane.followed_plane) == 0 and enemy_plane.ready_missile>0:
                                if TSVector3.distance(enemy_plane.pos3d, my_plane.pos3d) < dis:
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
                                if my_plane.ready_missile:
                                    far_away_dis = self.attack_distance
                                else:
                                    far_away_dis = my_plane.para["safe_range"]
                                if TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)<far_away_dis:
                                    if my_plane.move_speed > enemy_plane.Speed:
                                        my_plane.move_speed = max(enemy_plane.Speed-5,my_plane.para["move_min_speed"])
                                else:
                                    my_plane.move_speed = my_plane.para["move_max_speed"]
                                if 0<enemy_plane.lost_flag:
                                    cmd_list.append(env_cmd.make_linepatrolparam(my_plane.ID, [enemy_plane.pos3d,], my_plane.move_speed,
                                                            my_plane.para["move_max_acc"], my_plane.para["move_max_g"]))
                                else:
                                    enemy_plane.be_seen = True
                                    cmd_list.append(env_cmd.make_followparam(my_plane.ID, enemy_plane.ID, my_plane.move_speed, my_plane.para['move_max_acc'], my_plane.para['move_max_g']))   
                            else:
                                my_plane.follow_plane = None
                                print(my_plane.ID,my_plane.move_order,"跟踪敌机")
                    elif my_plane.follow_plane and my_plane.is_spy==False:
                        # 选择更近的飞机进行追踪
                        enemy_plane = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                        followed_plane_id = my_plane.ID
                        new_follow_plane = enemy_plane
                        for other_enemy_plane in self.enemy_plane: # 遇到更近的敌人转移
                            if other_enemy_plane.ID != new_follow_plane.ID:
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
                        if new_follow_plane.ID != enemy_plane.ID:
                            enemy_plane.followed_plane.remove(my_plane.ID)
                            new_follow_plane.followed_plane.append(my_plane.ID)
                            my_plane.follow_plane = new_follow_plane.ID
                            enemy_plane = new_follow_plane
                        if my_plane.ready_missile:
                            far_away_dis = self.attack_distance
                        else:
                            far_away_dis = my_plane.para["safe_range"]
                        if TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)<far_away_dis:
                            if my_plane.move_speed > enemy_plane.Speed:
                                my_plane.move_speed = max(enemy_plane.Speed-5, my_plane.para["move_min_speed"])
                        else:
                            my_plane.move_speed = my_plane.para["move_max_speed"]
                        if 0<enemy_plane.lost_flag:
                            cmd_list.append(env_cmd.make_linepatrolparam(my_plane.ID, [enemy_plane.pos3d,], my_plane.move_speed,
                                                    my_plane.para["move_max_acc"], my_plane.para["move_max_g"]))
                        else:
                            enemy_plane.be_seen = True
                            cmd_list.append(env_cmd.make_followparam(my_plane.ID, enemy_plane.ID, my_plane.move_speed, my_plane.para['move_max_acc'], my_plane.para['move_max_g']))

            # 追击敌方有人机模块 
            if len(self.free_plane)>0:
                for enemy_plane in self.enemy_leader_plane:
                    for followed_plane_id in enemy_plane.followed_plane.copy():
                        my_plane = self.get_body_info_by_id(self.my_plane, followed_plane_id)
                        if followed_plane_id in self.free_plane and my_plane.ready_missile > 0:
                            if self.is_in_center(my_plane, center_radius=140000) == False and len(enemy_plane.followed_plane)>1:
                                enemy_plane.followed_plane.remove(followed_plane_id)
                            else:
                                for other_enemy_plane in self.enemy_leader_plane: # 遇到更近的敌人转移
                                    if other_enemy_plane.ID != enemy_plane.ID:
                                        if TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d) > TSVector3.distance(my_plane.pos3d, other_enemy_plane.pos3d) + 800 \
                                                and followed_plane_id not in other_enemy_plane.followed_plane and other_enemy_plane.lost_flag==0:
                                            enemy_plane.followed_plane.remove(followed_plane_id)
                                            other_enemy_plane.followed_plane.append(followed_plane_id)
                                            my_plane.follow_plane = other_enemy_plane.ID
                                            break
                        elif followed_plane_id not in self.free_plane:
                            enemy_plane.followed_plane.remove(followed_plane_id)
                            my_plane.follow_plane = None
                for fre_ID in self.free_plane:
                    my_plane = self.get_body_info_by_id(self.my_plane, fre_ID)
                    if my_plane.Type == 1 and self.revenge == False or my_plane.is_spy==True:
                        continue
                    if my_plane.follow_plane is None:
                        dis = 9999999
                        for enemy_plane in self.enemy_leader_plane:
                            if len(enemy_plane.followed_plane)<=4:
                                if my_plane.ready_missile> 0:
                                    tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                                    if tmp_dis < dis and enemy_plane.lost_flag==0:
                                        my_plane.follow_plane = enemy_plane.ID
                                        dis = tmp_dis
                        if my_plane.follow_plane is None and my_plane.ready_missile == 0:
                            dis = 9999999
                            for enemy_plane in self.enemy_leader_plane:
                                tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                                if len(enemy_plane.followed_plane)==0 and enemy_plane.have_no_missile_plane == None:
                                    if tmp_dis < dis and enemy_plane.lost_flag==0:
                                        my_plane.follow_plane = enemy_plane.ID
                                        dis = tmp_dis
                            if my_plane.follow_plane is not None:
                                enemy = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                                enemy.have_no_missile_plane = my_plane.ID
                    if my_plane.follow_plane is not None:
                        enemy_plane = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                        if my_plane.ID not in enemy_plane.followed_plane and enemy_plane.lost_flag==0:
                            enemy_plane.followed_plane.append(my_plane.ID)
                        elif enemy_plane.lost_flag>0:
                            my_plane.follow_plane = None
                for enemy_plane in self.enemy_leader_plane:
                    followed_flag = 0
                    if enemy_plane.Availability:
                        if len(enemy_plane.followed_plane)>=2 and enemy_plane.have_no_missile_plane in enemy_plane.followed_plane:
                            enemy_plane.followed_plane.remove(enemy_plane.have_no_missile_plane)
                            follow_plane = self.get_body_info_by_id(self.my_plane,enemy_plane.have_no_missile_plane)
                            follow_plane.follow_plane = None
                            enemy_plane.have_no_missile_plane = None
                        enemy_plane.followed_plane = sorted(enemy_plane.followed_plane, key=lambda d: TSVector3.distance(self.get_body_info_by_id(self.my_plane,d).pos3d,enemy_plane.pos3d), reverse=True)
                        for plane_id in enemy_plane.followed_plane.copy():
                            my_plane = self.get_body_info_by_id(self.my_plane, plane_id)
                            if plane_id in self.free_plane and enemy_plane.lost_flag==0:
                                if my_plane.move_order==None:
                                    self.free_plane.remove(plane_id)
                                    my_plane.follow_plane = enemy_plane.ID
                                    followed_flag = 1
                                    my_plane.move_order="追踪敌机哈哈哈"
                                    my_plane.total_threat_flag = None
                                    dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                                    if my_plane.ready_missile:
                                        far_away_dis = self.attack_distance
                                    else:
                                        far_away_dis = my_plane.para["safe_range"]
                                    if dis > far_away_dis:
                                        my_plane.move_speed = my_plane.para['move_max_speed']
                                    else:
                                        if my_plane.move_speed > enemy_plane.Speed:
                                            my_plane.move_speed = max(enemy_plane.Speed-5,my_plane.para["move_min_speed"])
                                    cmd_list.append(env_cmd.make_followparam(my_plane.ID, enemy_plane.ID, my_plane.move_speed, my_plane.para['move_max_acc'], my_plane.para['move_max_g']))    
                                else:
                                    print(my_plane.ID,my_plane.move_order,"追踪敌机哈哈哈")
                        if followed_flag == 0:# 回到敌机上一次出现的位置搜查
                            dis = 999999
                            follow_plane = None
                            for fre_ID in self.free_plane:
                                my_plane = self.get_body_info_by_id(self.my_plane, fre_ID)
                                if my_plane.Type == 1 and self.revenge == False or my_plane.is_spy==True:
                                    continue
                                if my_plane.ready_missile> 0:
                                    tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                                    if tmp_dis < dis:
                                        dis = tmp_dis
                                        follow_plane = my_plane
                            if follow_plane is not None and enemy_plane.lost_flag<self.get_turn_time(follow_plane,enemy_plane)+2:
                                if follow_plane.move_order==None:
                                    self.free_plane.remove(follow_plane.ID)
                                    follow_plane.move_order="搜查敌有人机"
                                    follow_plane.total_threat_flag = None
                                    if follow_plane.ready_missile:
                                        far_away_dis = self.attack_distance
                                    else:
                                        far_away_dis = follow_plane.para["safe_range"]
                                    if TSVector3.distance(follow_plane.pos3d, enemy_plane.pos3d)<far_away_dis:
                                        if follow_plane.move_speed > enemy_plane.Speed:
                                            follow_plane.move_speed = max(enemy_plane.Speed-5,follow_plane.para["move_min_speed"])
                                    else:
                                        follow_plane.move_speed = follow_plane.para["move_max_speed"]
                                    cmd_list.append(env_cmd.make_linepatrolparam(follow_plane.ID, [enemy_plane.pos3d,], follow_plane.move_speed,
                                                follow_plane.para["move_max_acc"], follow_plane.para["move_max_g"]))
                                else:
                                    print(follow_plane.ID, follow_plane.move_order,"搜查敌有人机")
                            elif follow_plane is not None:
                                follow_plane.follow_plane = None
                        
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
                        print(plane.name,'僚机不匹配')
                if plane.wing_plane!=None:
                    wing_plane = self.get_body_info_by_id(self.my_plane, plane.wing_plane)
                    if wing_plane.wing_who!=plane.ID:
                        print(plane.name,"僚机不匹配")
    
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
                        if i==1 or i==3:
                            self.spy_plane.append(plane.ID)
                            plane.is_spy = True
                        cmd_list.append(
                            env_cmd.make_entityinitinfo(plane.ID, -125000 * self.side, 150000 - (i+1) * 50000, 9000, 300, self.init_direction))
                    else:
                        self.spy_plane.append(plane.ID)
                        plane.is_spy = True
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
        cmd_list.append(env_cmd.make_linepatrolparam(leader_plane_1.ID, route_list, 250, 1, 6))

        new_pos = TSVector3.plus(leader_plane_2.pos3d, TSVector3.multscalar(new_dir, distance))
        new_pos['Z'] = 9000
        route_list = [new_pos]
        cmd_list.append(env_cmd.make_linepatrolparam(leader_plane_2.ID, route_list, 250, 1, 6))

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
            if len(self.enemy_leader_plane) and self.is_in_center(plane, center_radius=40000)==False:
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
                if plane.is_spy:# 暗度陈仓
                    if len(self.enemy_leader_plane):
                        dis = 99999999
                        spy_enemy = None
                        for enemy in self.enemy_leader_plane:
                            if TSVector3.distance(enemy.pos3d, plane.pos3d)<dis and enemy.Availability:
                                dis = TSVector3.distance(enemy.pos3d, plane.pos3d)
                                spy_enemy = enemy
                        if spy_enemy!=None:
                            new_dir = TSVector3.calorientation(spy_enemy.Heading+math.pi, relative_pitch)
                            new_pos = TSVector3.plus(spy_enemy.pos3d, TSVector3.multscalar(new_dir, spy_enemy.Speed*6))
                            if (self.side==1 and plane.X>5000) or (self.side==-1 and plane.X<-5000):
                                pass
                            else:
                                new_pos['Z'] = 3000
                            route_list = [new_pos,]
                    else:
                        target_pos = {"X": 135000 * self.side, "Y": (random.random()*2-1)*3000, "Z": 2000}
                        new_heading = TSVector3.calheading(TSVector3.minus(target_pos,plane.pos3d))
                        # new_heading = init_direction/180*math.pi
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
                    if enemy_can_see_me==False or distance<6000:
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
                if (enemy_can_see_me == False and my_plane.Type==1) or my_plane.Type==2 or (self.my_score>self.enemy_score+self.win_score) or enemy_plane.be_seen_leader:
                    can_attack_dict[attack_plane.ID] = enemy_plane.ID

        return can_attack_dict

    # 实施干扰
    def activate_jam(self, cmd_list):
        leader1 = self.my_leader_plane[0]
        leader2 = self.my_leader_plane[1]
        # 找到最中间的敌机进行干扰
        for leader in self.my_leader_plane:
            middle_enemy_dict = {}
            for enemy in self.enemy_plane:
                if enemy.lost_flag==0 and enemy.can_see(leader):
                    dis =  leader.pi_bound(TSVector3.calheading(TSVector3.minus(enemy.pos3d,leader.pos3d))-leader.Heading)
                    if enemy.ID not in middle_enemy_dict:
                        middle_enemy_dict[enemy.ID] = dis
            middle_enemy_list = [key for key, value in sorted(middle_enemy_dict.items(), key=lambda d: d[1])]
            if len(middle_enemy_list):
                leader.middle_enemy_plane = middle_enemy_list[int(len(middle_enemy_list)/2)]
                leader.left_enemy_plane = middle_enemy_list[0]
                leader.right_enemy_plane = middle_enemy_list[-1]
            else:
                leader.middle_enemy_plane = None

        need_jam_list = []
        evade_list = []
        for i, plane in enumerate(self.my_leader_plane):
            if plane.Availability==0:
                continue
            if 60>self.sim_time - plane.last_jam>6:
                plane.do_turn==False
                plane.do_jam = False
            elif self.sim_time - plane.last_jam<7:
                plane.do_turn = False
                plane.do_jam = True
            else:
                if plane.do_turn==False:
                    plane.do_jam = False
            check_turn = False
            if self.my_leader_plane[1-i].Availability and self.sim_time-self.my_leader_plane[1-i].last_jam>60\
                  and self.my_leader_plane[1-i].do_jam==False and self.check_jam_after_turn(plane, self.my_leader_plane[1-i]):
                check_turn = True
            if self.sim_time-plane.last_jam>60 and plane.do_jam==False and self.check_jam_after_turn(plane, plane):
                check_turn = True
            if check_turn:
                need_jam_list.append(plane.ID)
            if len(plane.locked_missile_list)>0:
                evade_list.append(plane.ID)

        if len(need_jam_list):
            if leader1.ID not in need_jam_list and leader1.Availability:# 1号有人机空闲，2号有人机需要干扰支持
                other_can_save_me = False
                if self.sim_time-leader1.last_jam>60 and self.sim_time - leader2.jammed>=6:
                    if self.check_jam_after_turn(leader2, leader1):# 有人机1需要进行转弯判断
                        if self.plane_can_jam(leader2, leader1, cmd_list) == False:
                            self.find_way_jam(leader1, leader2.middle_enemy_plane ,cmd_list)
                        other_can_save_me = True
                if other_can_save_me==False:
                    self.save_myself(leader2, need_jam_list, cmd_list)

            elif leader2.ID not in need_jam_list and leader2.Availability:# 2号有人机空闲，1号有人机需要干扰支持
                other_can_save_me = False
                if self.sim_time-leader1.last_jam>60 and self.sim_time - leader1.jammed>=6:
                    if self.check_jam_after_turn(leader1,leader2):# 有人机2需要进行转弯判断
                        if self.plane_can_jam(leader1, leader2, cmd_list) == False:
                            self.find_way_jam(leader2, leader1.middle_enemy_plane, cmd_list)
                        other_can_save_me = True
                if other_can_save_me==False:
                    self.save_myself(leader1, need_jam_list, cmd_list)
            else:
                # 1号、2号有人机需要干扰支持，各自顾自己
                if leader1 in need_jam_list:
                    save_myself1 = self.save_myself(leader1, need_jam_list, cmd_list)  
                elif leader1 in evade_list:
                    save_myself1 = False
                elif leader1.Availability:
                    save_myself1 = True  
                else:
                    save_myself1 = False
                if leader2 in need_jam_list:
                    save_myself2 = self.save_myself(leader2, need_jam_list, cmd_list) 
                elif leader2 in evade_list:
                    save_myself2 = False
                elif leader2.Availability:
                    save_myself2 = True  
                else:
                    save_myself2 = False
                if save_myself1==False:# 救不了自己，别人也救不了他自己，看看能否互救对方
                    self.plane_can_jam(leader1,leader2,cmd_list)   
                if save_myself2==False:
                    self.plane_can_jam(leader2,leader1,cmd_list)
        if len(evade_list):
            for leader_id in evade_list:
                leader = self.get_body_info_by_id(self.my_plane, leader_id)
                if leader.do_turn or (leader.do_jam and TSVector3.distance(leader.pos3d, leader.close_missile.pos3d)>10000):
                    continue
                total_dir,min_dis,threat_pitch = self.synthetic_threat_vector(leader, see_dis=leader.para["safe_range"],seen=False)
                new_plane_pos = leader.evade_leader(leader.close_missile, total_dir, cmd_list)
                if leader.close_missile and (leader.close_missile.init_dis<6000 or self.death_to_death(leader,leader.close_missile,new_plane_pos)):
                    self.all_death(leader,cmd_list)
        
    # 有人机的干扰自顾自
    def save_myself(self, leader, need_jam_list, cmd_list):
        if leader.ID in need_jam_list and leader.do_jam==False:
            enemy = self.get_body_info_by_id(self.enemy_plane, leader.middle_enemy_plane)
            can_jam = False
            check_turn = self.check_jam_after_turn(leader,leader)# 有人机需要进行转弯判断
            if self.sim_time-leader.last_jam>60  and self.sim_time - leader.jammed>6 and enemy is not None and check_turn:
                if enemy.lost_flag==0 and leader.can_see(enemy, see_factor=0.6, jam_dis=117600):
                    leader.do_jam = True
                    leader.do_turn = False
                    cmd_list.append(env_cmd.make_jamparam(leader.ID))
                    leader.jammed = self.sim_time+1
                    leader.last_jam = self.sim_time+7
                    can_jam = True
                elif TSVector3.distance(leader.pos3d, leader.close_missile.pos3d)/1000 > 11 + self.get_turn_time(leader, enemy, see_factor=0.6):
                    self.find_way_jam(leader, leader.middle_enemy_plane, cmd_list)
                    can_jam = True
                else:
                    can_jam = False
            if can_jam == False:
                leader.do_jam = False
                leader.do_turn = False
                return False
            else:
                return True

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

    # 判断有人机是否可以进行干扰
    def plane_can_jam(self, need_jam_plane, jam_plane, cmd_list):
        if need_jam_plane.middle_enemy_plane is not None:
            enemy = self.get_body_info_by_id(self.enemy_plane, need_jam_plane.middle_enemy_plane)
            if enemy.lost_flag==0:
                can_jam = False
                # 别人可以帮忙吗
                if not need_jam_plane.can_see(enemy, see_factor=0.6,jam_dis=117600) and jam_plane.can_see(enemy, see_factor=0.9,jam_dis=117600):
                    if len(jam_plane.locked_missile_list)==0:
                        if jam_plane.move_order==None:
                            jam_plane.move_order="干扰0"
                            jam_plane.total_threat_flag = None
                            cmd_list.append(env_cmd.make_followparam(jam_plane.ID, enemy.ID, jam_plane.move_speed, jam_plane.para['move_max_acc'], jam_plane.para['move_max_g']))
                            can_jam = True 
                        else:
                            print(jam_plane.ID,jam_plane.move_order,"干扰0")
                            
                    elif jam_plane.can_see(enemy, see_factor=0.6, jam_dis=117600):
                        can_jam = True
                    else:
                        can_jam = False
                # 靠自己看行不行
                elif need_jam_plane.can_see(enemy, see_factor=0.6,jam_dis=117600) and self.sim_time-need_jam_plane.last_jam>60 \
                            and self.sim_time - need_jam_plane.jammed>6 \
                            and self.check_jam_after_turn(need_jam_plane,need_jam_plane):# 有人机需要进行转弯判断
                    need_jam_plane.do_jam = True
                    need_jam_plane.do_turn = False
                    cmd_list.append(env_cmd.make_jamparam(need_jam_plane.ID))
                    need_jam_plane.jammed = self.sim_time+1
                    need_jam_plane.middle_enemy_plane = None
                    need_jam_plane.last_jam = self.sim_time+7
                    return True
                if can_jam:
                    jam_plane.do_jam = True
                    jam_plane.do_turn = False
                    cmd_list.append(env_cmd.make_jamparam(jam_plane.ID))
                    need_jam_plane.jammed = self.sim_time+1
                    need_jam_plane.middle_enemy_plane = None
                    jam_plane.last_jam = self.sim_time+7
                    return True
        return False
    
    # 判断有人机转弯之后是否还有干扰的必要
    def check_jam_after_turn(self,need_jam_plane,jam_plane,check_dis=21950):
        middle_enemy_id = need_jam_plane.middle_enemy_plane
        if middle_enemy_id == None:
            return False
        left_enemy_plane = self.get_body_info_by_id(self.enemy_plane, need_jam_plane.left_enemy_plane)
        right_enemy_plane = self.get_body_info_by_id(self.enemy_plane, need_jam_plane.right_enemy_plane)
        if abs(jam_plane.pi_bound(TSVector3.calheading(TSVector3.minus(right_enemy_plane.pos3d, jam_plane.pos3d))
                                  -TSVector3.calheading(TSVector3.minus(left_enemy_plane.pos3d, jam_plane.pos3d))))>jam_plane.para['radar_heading']*2*math.pi/180:
            return False            
        jam_enemy = self.get_body_info_by_id(self.enemy_plane, middle_enemy_id)
        turn_time = self.get_turn_time(jam_plane,jam_enemy,see_factor=0.9)
        can_jam_after_turn = False
        if jam_plane.close_missile:
            if TSVector3.distance(jam_plane.pos3d, jam_plane.close_missile.pos3d)<1000*(turn_time+5)+11000:
                return False
        for missile_id in need_jam_plane.locked_missile_list:
            missile = self.get_body_info_by_id(self.missile_list,missile_id)
            if self.sim_time-missile.fire_time<10 and missile.missile_acc!=0 and missile.Speed<1000:
                acc_t = (1000-missile.Speed)/missile.missile_acc
                if acc_t > (turn_time+5):
                    acc_dis = (turn_time+5)*(turn_time+5)*0.5*missile.missile_acc
                else:
                    acc_dis = acc_t*acc_t*0.5*missile.missile_acc+ (turn_time+5 - acc_t)*(1000-missile.Speed)
                if TSVector3.distance(missile.pos3d,need_jam_plane.pos3d)>check_dis+(missile.Speed)*(turn_time+5) + acc_dis:
                    can_jam_after_turn = True
            else:
                if TSVector3.distance(missile.pos3d,need_jam_plane.pos3d)>check_dis+(missile.Speed+missile.missile_acc)*(turn_time+5):
                    can_jam_after_turn = True
        return can_jam_after_turn

    # 跟踪可以进行干扰但不在探测范围的敌机
    def find_way_jam(self, jam_plane, middle_enemy_id, cmd_list):
        if middle_enemy_id==None:
            return
        jam_enemy = self.get_body_info_by_id(self.enemy_plane, middle_enemy_id)
        if jam_enemy.lost_flag==0:
            if jam_plane.move_order==None:
                jam_plane.do_jam = True
                jam_plane.do_turn = True
                jam_plane.total_threat_flag = None
                jam_plane.move_order="搜寻干扰1"
                cmd_list.append(env_cmd.make_followparam(jam_plane.ID, jam_enemy.ID, jam_plane.move_speed, jam_plane.para['move_max_acc'], jam_plane.para['move_max_g']))
            else:
                print(jam_plane.ID,jam_plane.move_order,"搜寻干扰1")
        else:
            if jam_plane.move_order==None:
                jam_plane.do_jam = True
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
        if self.sim_time>19*60+57:
            print("studio", self.my_score ,self.enemy_score, self.my_center_time, self.enemy_center_time)
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
        min_dis = 9999999
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
                            tmp_dir2 = TSVector3.multscalar(TSVector3.minus(plane.pos3d, leader.pos3d), ratio*math.pow(leader.para["safe_range"]-1000/(dis+200),2))
                            threat_weight += ratio*math.pow(leader.para["safe_range"]-1000/(dis+200),2)
                            total_dir = TSVector3.plus(tmp_dir2,total_dir)
        elif len(self.enemy_plane)>5 and leader.ready_missile==0 and total_dir == {"X": 0, "Y": 0, "Z": 0}:
            for plane in self.my_plane:
                if plane.Availability and plane.ID != leader.ID and plane.ready_missile:
                    dis = TSVector3.distance(plane.pos3d, leader.pos3d)
                    if dis<60000:
                        tmp_dir2 = TSVector3.multscalar(TSVector3.minus(plane.pos3d, leader.pos3d), math.pow(leader.para["safe_range"]-1000/(dis+200),2))
                        threat_weight += math.pow(leader.para["safe_range"]-1000/(dis+200),2)
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