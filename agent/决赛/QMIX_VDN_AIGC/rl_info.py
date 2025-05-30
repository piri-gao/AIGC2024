import math
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.agent import Agent
import numpy as np
from agent.QMIX_VDN_AIGC.agent_base import Plane, Missile
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
        # 每个智能体的奖励
        self.rewards = [0 for i in range(10)]
        # 智能体done信息
        self.done = [0 for i in range(10)]
        # 我方空闲飞机
        self.free_plane = []
        # 下标与ID转换
        self.index_to_id = {}
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
            plane.be_seen_myleader = False
            plane.be_seen = False
            plane.be_in_danger = False
            if self.sim_time<20:
                plane.jammed = -100
            if plane.Type==1 and self.sim_time-plane.last_jam>0:
                plane.do_jam = False
            if plane.lost_flag and plane.Availability:
                plane.Availability = 0
                self.done[i] = 1
                self.rewards[i] -= 2 if plane.Type==2 else 4
            elif plane.Availability and self.is_in_center(plane):
                plane.center_time += 1
                self.rewards[i] += 0.8
            elif plane.Availability:
                self.rewards[i] += 0.5
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                    self.rewards[i] -= 0.1
                if missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list and missile.lost_flag==0 and missile.loss_target==False:
                    plane.locked_missile_list.append(missile.ID)
                    self.rewards[i] -= 0.02
                if missile.EngageTargetID == plane.ID and missile.loss_target==False:
                    short_dis, missile.arrive_time = self.shortest_distance_between_linesegment(plane, missile)
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
                        if tmp_missile.lost_flag and self.death_to_death(plane,tmp_missile,next_plane_pos,radius=100):
                            dead_flag += 1
                            attack_plane = self.get_body_info_by_id(self.my_plane, tmp_missile.LauncherID)
                            self.rewards[attack_plane.i] += 2 if plane.Type==2 else 4
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
            print("预测赢了:",self.my_score,'   ',self.enemy_score)
            for i in range(len(self.rewards)):
                self.rewards[i] += 4
        elif self.sim_time>19*60+57 and self.my_score<self.enemy_score:
            for i in range(len(self.rewards)):
                self.rewards[i] -= 0.7
        if self.sim_time>19*60+57:
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
            if have_enemy and (self.enemy_left_weapon < 2 or (self.sim_time > 20 * 60 -300\
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
            self.bound = 144000
    
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
    
    # 计算是否逃不掉了
    def death_to_death(self, plane,  missile, new_plane_pos=None, radius = 180):
        dis, t = self.shortest_distance_between_linesegment(plane, missile, new_plane_pos)
        if dis<radius:
            return True
        else:
            return False
    
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
            lost_flag = np.array([missile.lost_flag])
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
                    lost_flag = np.array([plane.lost_flag])
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
                    lost_flag = np.array([cur_plane.lost_flag])
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
            action_space = 8
            action_mask = np.ones(action_space)
            plane = self.get_body_info_by_id(self.my_plane, self.index_to_id[i])
            if plane.Type==2:
                action_mask[7] = 0
            elif self.sim_time - plane.last_jam<60:
                action_mask[7] = 0
            else:
                have_enemy_jam = False
                for enemy in self.enemy_plane:
                    if plane.can_see(enemy,see_factor=0.9):
                        have_enemy_jam = True
                if have_enemy_jam==False:
                    action_mask[7] = 0
            have_enemy_attack = False
            for enemy in self.enemy_plane:
                if enemy.lost_flag==0 and plane.can_attack(enemy) and plane.ready_missile>0:
                    have_enemy_attack = True
                    break
            if have_enemy_attack==False:
                action_mask[5] = 0
            have_enemy_follow = False
            for enemy in self.enemy_plane:
                if enemy.lost_flag==0:
                    have_enemy_follow = True
            if have_enemy_follow==False:
                action_mask[6] = 0
            actions_mask.append(action_mask)
        actions_mask = np.array(actions_mask)
        return actions_mask
    
    # 选择模块
    def selected_module(self, action):
        MODULE = {
            0: self.go_ahead,
            1: self.left_turn,
            2: self.right_turn,
            3: self.up_sky,
            4: self.down_ground,
            5: self.attack_enemy,
            6: self.follow_enemy,
            7: self.activate_jam
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
    
    # 根据动作映射至命令
    def action2order(self, action, cmd_list):
        for i, act in enumerate(action):
            plane = self.get_body_info_by_id(self.my_plane, self.index_to_id[i])
            self.selected_module(act)(cmd_list, plane)

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
        
    # 判断是否赢了       
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