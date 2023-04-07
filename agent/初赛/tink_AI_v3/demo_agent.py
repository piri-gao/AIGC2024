import copy
from typing import List
import numpy as np
import math
import random
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.agent import Agent
from agent.tink_AI_v3.agent_base import Plane, Missile

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
        # 干扰列表
        self.need_jam_list = []
        # 当前时间
        self.sim_time = 0

    def reset(self):
        self._init()

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        cmd_list = []
        self.update_decision(sim_time, obs_side, cmd_list)
        return cmd_list

    def update_dead_info(self, sim_time):
        for plane in self.my_plane:
            if plane.lost_flag:
                plane.Availability = 0
                if plane.follow_plane is not None:
                    enemy = self.get_body_info_by_id(self.enemy_plane, plane.follow_plane)
                    if plane.ID in enemy.followed_plane:
                        enemy.followed_plane.remove(plane.ID)
                        plane.follow_plane = None
            elif self.is_in_center(plane):
                plane.center_time += 1
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                if missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list and missile.lost_flag==0:
                    plane.locked_missile_list.append(missile.ID)
            # 更新导弹追踪信息
            for missile_id in plane.locked_missile_list.copy():
                tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if tmp_missile.lost_flag and sim_time - tmp_missile.arrive_time > 2:
                    plane.locked_missile_list.remove(tmp_missile.ID)
            plane.ready_missile = plane.LeftWeapon - len(plane.used_missile_list)

        for plane in self.enemy_plane:
            if plane.lost_flag: 
                if len(plane.locked_missile_list)>0:
                    dead_flag = 1
                    for missile_id in plane.locked_missile_list:
                        tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                        if tmp_missile.lost_flag==0:
                            dead_flag = 0
                    if dead_flag:
                        plane.Availability = 0
            else:
                if self.is_in_center(plane):
                    plane.center_time += 1
                plane.Availability = 1
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                if missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list and missile.lost_flag==0:
                    plane.locked_missile_list.append(missile.ID)
            # 更新导弹追踪信息
            for missile_id in plane.locked_missile_list.copy():
                tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if tmp_missile.lost_flag:
                    plane.locked_missile_list.remove(tmp_missile.ID)
            plane.ready_missile = plane.LeftWeapon - len(plane.used_missile_list)

        # 更新是否反击
        if self.revenge == False:
            self.first_formation = self.my_uav_plane
            enemy_left_weapon = 24
            have_enemy = 0
            self.my_left_weapon = 16
            for plane in self.my_uav_plane:
                if plane.Availability:
                    self.my_left_weapon -= len(plane.used_missile_list)
                else:
                    self.my_left_weapon -= plane.LeftWeapon
            for plane in self.enemy_plane:
                if plane.Availability:
                    have_enemy = 1
                    enemy_left_weapon -= len(plane.used_missile_list)
                    if plane.Type == 1:
                        for my_plane in self.my_leader_plane:
                            if my_plane.can_see(plane, see_factor=0.95):
                                self.revenge = True
                else:
                    enemy_left_weapon -= plane.LeftWeapon
            if have_enemy and enemy_left_weapon < 5 or sim_time > 20 * 60 -300 or self.my_left_weapon == 0:
                self.revenge = True

        if self.revenge:
            self.first_formation = self.my_plane

    def update_entity_info(self, obs_side, sim_time):
        # 己方所有飞机信息
        self.my_plane = self.get_all_body_list(self.my_plane, obs_side['platforminfos'], Plane)
        # 敌方所有飞机信息
        self.enemy_plane = self.get_all_body_list(self.enemy_plane, obs_side['trackinfos'], Plane)
        # 获取双方导弹信息
        self.missile_list = self.get_all_body_list(self.missile_list, obs_side['missileinfos'], Missile)
        # 更新阵亡飞机
        self.update_dead_info(sim_time)
        # 己方有人机信息
        self.my_leader_plane = self.get_body_info_by_type(self.my_plane, 1)
        # 己方无人机信息
        self.my_uav_plane = self.get_body_info_by_type(self.my_plane, 2)
        # 敌方有人机信息
        self.enemy_leader_plane = self.get_body_info_by_type(self.enemy_plane, 1)
        # 敌方无人机信息
        self.enemy_uav_plane = self.get_body_info_by_type(self.enemy_plane, 2)
        # 获取队伍标识
        if self.side is None:
            if self.my_plane[0].Identification == "红方":
                self.side = 1
            else:
                self.side = -1
            self.init_direction = 90
            if self.side == -1:
                self.init_direction = 270 
        if len(self.enemy_leader_plane):
            self.bound = 40000 
        else:
            self.bound = 90000
        

    def get_all_body_list(self, agent_info_list, obs_list, cls):
        # 更新真实智能体信息
        for obs_agent_info in obs_list:
            agent_in_list = False
            for agent_info in agent_info_list:
                if obs_agent_info['ID'] == agent_info.ID:
                    agent_info.update_agent_info(obs_agent_info)
                    agent_in_list = True
                    break
            if not agent_in_list:
                if obs_agent_info['Type'] == 3:
                    if obs_agent_info['Identification'] != self.my_plane[0].Identification:
                        attacked_plane = self.get_body_info_by_id(self.my_plane, obs_agent_info['EngageTargetID'])
                    else:
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
        return agent_info_list

    # 通过一种类型获取实体信息
    def get_body_info_by_type(self, agent_info_list, agent_type: int) -> list:
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

    def formation(self):
        # 干扰
        self.jam_list = [[plane, 0, 0] for plane in self.my_leader_plane]

    def update_decision(self, sim_time, obs_side, cmd_list):
        self.update_entity_info(obs_side,sim_time)
        if sim_time <= 2:
            self.init_pos(sim_time, cmd_list)
        else:
            # 开火模块
            need_jam_list = []
            evade_plane_id = []
            free_plane = []
            no_missile_plane = []
            for my_plane in self.my_plane:
                if len(my_plane.locked_missile_list)==0 and my_plane.Availability:
                    free_plane.append(my_plane)
                    if my_plane.ready_missile == 0 and my_plane.Type == 2:
                        no_missile_plane.append(my_plane.ID)
                elif len(my_plane.locked_missile_list)>0 and my_plane.Availability:
                    evade_plane_id.append(my_plane.ID)
                    if my_plane.Type == 1:
                        for missile_id in my_plane.locked_missile_list:
                            missile = self.get_body_info_by_id(self.missile_list, missile_id)
                            if 20000 < TSVector3.distance(missile.pos3d, my_plane.pos3d) < 23000:
                                need_jam_list.append({"missile_entity":missile_id})
            free_plane = sorted(free_plane, key=lambda d: d.ready_missile, reverse=False)
            free_plane = [plane.ID for plane in sorted(free_plane, key=lambda d: d.Type, reverse=True)]
            # 指定僚机
            if len(self.enemy_plane):
                # dead = []
                # for plane in self.enemy_plane:
                #     if plane.Availability==0:
                #         dead.append(plane.ID)
                # if dead:
                #     print(sim_time, dead)
                for leader_plane in self.my_leader_plane:
                    if leader_plane.Availability:
                        if leader_plane.wing_plane != None:
                            wing_plane = self.get_body_info_by_id(self.my_plane, leader_plane.wing_plane)
                            if wing_plane.ready_missile!=0 and len(no_missile_plane)>0 or wing_plane.ID not in free_plane:
                                leader_plane.wing_plane = None
                        if leader_plane.wing_plane is None:
                            dis = 9999999999
                            for plane_id in no_missile_plane:
                                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                                if TSVector3.distance(plane.pos3d, leader_plane.pos3d) < dis and plane.Type==2:
                                    leader_plane.wing_plane = plane_id
                                    dis = TSVector3.distance(plane.pos3d, leader_plane.pos3d)
                            if leader_plane.wing_plane is not None:
                                no_missile_plane.remove(leader_plane.wing_plane)
                                free_plane.remove(leader_plane.wing_plane)
                            else:
                                dis = 9999999999
                                have_missile = 3
                                for plane_id in free_plane:
                                    plane = self.get_body_info_by_id(self.my_plane, plane_id)
                                    if plane.Type == 2 and TSVector3.distance(plane.pos3d, leader_plane.pos3d) < dis and plane.ready_missile <= have_missile:
                                        dis = TSVector3.distance(plane.pos3d, leader_plane.pos3d)
                                        have_missile = plane.ready_missile
                                        leader_plane.wing_plane = plane_id
                                if leader_plane.wing_plane is not None:
                                    free_plane.remove(leader_plane.wing_plane)
                         
            attack_enemy = {}
            for threat_plane in self.enemy_leader_plane:
                if threat_plane.Availability:
                    threat_ID = threat_plane.ID
                    attack_plane = self.can_attack_plane(threat_plane, evade_plane_id, free_plane)
                    for missile_id in threat_plane.locked_missile_list:
                        missile = self.get_body_info_by_id(self.missile_list, missile_id)
                        if threat_ID in attack_enemy:
                            attack_enemy[threat_ID] = max(missile.fire_time, attack_enemy[threat_ID])
                        else:
                            attack_enemy[threat_ID] = missile.fire_time
                    if attack_plane is not None:
                        can_attack_now = True
                        if threat_ID in attack_enemy.keys() and sim_time - attack_enemy[threat_ID] < 2:
                            can_attack_now = False
                        if can_attack_now:
                            factor_fight = 1
                            cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_ID, factor_fight))
                            attack_plane.ready_missile -= 1
            
            # 有人机脱离战场
            for leader in self.my_leader_plane:
                need_wing = False
                if leader.ID in free_plane and (self.my_left_weapon >2 or len(leader.used_missile_list)==4):
                    total_dir = {"X": 0, "Y": 0, "Z": 0}
                    min_dis = 9999999
                    for enemy in self.enemy_plane:
                        if enemy.Availability and (enemy.LeftWeapon - len(enemy.used_missile_list))>0:
                            if enemy.can_see(leader) or TSVector3.distance(enemy.pos3d, leader.pos3d)<45000:
                                min_dis = min(TSVector3.distance(enemy.pos3d, leader.pos3d), min_dis)
                                tmp_dir2 = TSVector3.minus(leader.pos3d, enemy.pos3d)
                                total_dir = TSVector3.plus(tmp_dir2,total_dir)
                    if total_dir != {"X": 0, "Y": 0, "Z": 0}:
                        next_heading = TSVector3.calheading(total_dir) + math.pi
                        if self.is_in_center(leader,100000) == False:
                            tmp_dir2 = {"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000}
                            tmp_dir2 = TSVector3.minus(tmp_dir2, leader.pos3d)
                            total_dir = TSVector3.plus(tmp_dir2, total_dir)
                            min_dis = 99999999
                        free_plane.remove(leader.ID)
                        total_heading = TSVector3.calheading(total_dir)
                        total_pitch = TSVector3.calpitch(total_dir)
                        if min_dis>30000 and min_dis!=99999999:
                            if abs(leader.pi_bound(leader.Heading - total_heading - math.pi/2))>math.pi/2:
                                total_heading = total_heading - math.pi/2
                            else:
                                total_heading = total_heading + math.pi/2
                        total_dir = TSVector3.calorientation(total_heading, total_pitch)
                        total_dir = TSVector3.normalize(total_dir)
                        distance = 15 * leader.para["move_max_speed"]
                        evade_pos = TSVector3.plus(leader.pos3d, TSVector3.multscalar(total_dir, distance))
                        straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": evade_pos['Z']}, ]
                        cmd_list.append(env_cmd.make_linepatrolparam(leader.ID, straight_evade_route_list,
                                            leader.para["move_max_speed"], leader.para["move_max_acc"], leader.para["move_max_g"]))
                        if leader.Availability:
                            wing_plane = self.get_body_info_by_id(self.my_plane, leader.wing_plane)
                            if wing_plane.ID in free_plane:
                                free_plane.remove(wing_plane.ID)
                            if TSVector3.distance(wing_plane.pos3d, leader.pos3d)>35000:
                                cmd_list.append(env_cmd.make_followparam(wing_plane.ID, leader.ID, wing_plane.para["move_max_speed"], wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
                            else:
                                total_dir = TSVector3.calorientation(next_heading, 0)
                                total_dir = TSVector3.normalize(total_dir)
                                distance = 15 * wing_plane.para["move_max_speed"]
                                next_pos = TSVector3.plus(wing_plane.pos3d, TSVector3.multscalar(total_dir, distance))
                                next_pos['Z'] = leader.Z
                                cmd_list.append(env_cmd.make_linepatrolparam(wing_plane.ID, [next_pos,],
                                            wing_plane.para["move_max_speed"], wing_plane.para["move_max_acc"], wing_plane.para["move_max_g"]))
                    else:
                        need_wing = True
                else:
                    need_wing = True
                if leader.Availability and need_wing:
                    if leader.wing_plane:
                        wing_plane = self.get_body_info_by_id(self.my_plane, leader.wing_plane)
                        if wing_plane.ID in free_plane:
                            free_plane.remove(wing_plane.ID)
                        if len(leader.locked_missile_list) > 0:
                            next_pos = {"X": 0, "Y": 0, "Z": 0}
                            dis = 9999999
                            for missile_id in leader.locked_missile_list:
                                missile = self.get_body_info_by_id(self.missile_list, missile_id)
                                if TSVector3.distance(leader.pos3d, missile.pos3d) < dis and sim_time - missile.arrive_time<-3:
                                    dis = TSVector3.distance(leader.pos3d, missile.pos3d)
                                    next_pos = missile.pos3d
                            if next_pos != {"X": 0, "Y": 0, "Z": 0}:
                                cmd_list.append(env_cmd.make_linepatrolparam(wing_plane.ID, [next_pos,],
                                            wing_plane.para["move_max_speed"], wing_plane.para["move_max_acc"], wing_plane.para["move_max_g"]))
                            else:
                                cmd_list.append(env_cmd.make_followparam(wing_plane.ID, leader.ID, wing_plane.para["move_max_speed"], wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
                        else:
                            cmd_list.append(env_cmd.make_followparam(wing_plane.ID, leader.ID, wing_plane.para["move_max_speed"], wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
            
            # 追击敌方有人机模块 
            if len(free_plane)>0:
                for enemy_plane in self.enemy_leader_plane:
                    for followed_plane_id in enemy_plane.followed_plane.copy():
                        my_plane = self.get_body_info_by_id(self.my_plane, followed_plane_id)
                        if followed_plane_id in free_plane and my_plane.ready_missile > 0:
                            if self.is_in_center(my_plane, center_radius=44000) == False:
                                enemy_plane.followed_plane.remove(followed_plane_id)
                            else:
                                for other_enemy_plane in self.enemy_leader_plane: # 遇到更近的敌人转移
                                    if other_enemy_plane.ID != enemy_plane.ID:
                                        if TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d) > TSVector3.distance(my_plane.pos3d, other_enemy_plane.pos3d) + 800 and followed_plane_id not in other_enemy_plane.followed_plane:
                                            enemy_plane.followed_plane.remove(followed_plane_id)
                                            other_enemy_plane.followed_plane.append(followed_plane_id)
                                            my_plane.follow_plane = other_enemy_plane.ID
                                            break
                        else:
                            enemy_plane.followed_plane.remove(followed_plane_id)
                            my_plane.follow_plane = None
                for fre_ID in free_plane:
                    my_plane = self.get_body_info_by_id(self.my_plane, fre_ID)
                    if my_plane.Type == 1 and self.revenge == False:
                        continue
                    if my_plane.follow_plane is None:
                        dis = 9999999
                        for enemy_plane in self.enemy_leader_plane:
                            if len(enemy_plane.followed_plane)<=4:
                                if my_plane.ready_missile> 0:
                                    tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                                    if tmp_dis < dis and self.is_in_center(my_plane, center_radius=44000)==True:
                                        my_plane.follow_plane = enemy_plane.ID
                                        dis = tmp_dis
                        if my_plane.follow_plane is None and my_plane.ready_missile == 0:
                            dis = 9999999
                            for enemy_plane in self.enemy_leader_plane:
                                tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                                if len(enemy_plane.followed_plane)==0 and enemy_plane.have_no_missile_plane == None:
                                    if tmp_dis < dis and self.is_in_center(my_plane, center_radius=44000)==True:
                                        my_plane.follow_plane = enemy_plane.ID
                                        dis = tmp_dis
                            if my_plane.follow_plane is not None:
                                enemy = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                                enemy.have_no_missile_plane = my_plane.ID
                    if my_plane.follow_plane is not None:
                        enemy_plane = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                        if my_plane.ID not in enemy_plane.followed_plane:
                            enemy_plane.followed_plane.append(my_plane.ID)
                for enemy_plane in self.enemy_leader_plane:
                    followed_flag = 0
                    if enemy_plane.Availability:
                        if len(enemy_plane.followed_plane)>=2 and enemy_plane.have_no_missile_plane in enemy_plane.followed_plane:
                            enemy_plane.followed_plane.remove(enemy_plane.have_no_missile_plane)
                            enemy_plane.have_no_missile_plane = None

                        for plane_id in enemy_plane.followed_plane.copy():
                            my_plane = self.get_body_info_by_id(self.my_plane, plane_id)
                            if self.is_in_center(my_plane, center_radius=35000):
                                if plane_id in free_plane:
                                    free_plane.remove(plane_id)
                                    my_plane.follow_plane = enemy_plane.ID
                                    followed_flag = 1
                                    cmd_list.append(env_cmd.make_followparam(my_plane.ID, enemy_plane.ID, my_plane.para["move_max_speed"], my_plane.para['move_max_acc'], my_plane.para['move_max_g']))
                            else:
                                if len(enemy_plane.followed_plane)>1:
                                    enemy_plane.followed_plane.remove(plane_id)
                                    my_plane.follow_plane = None
                                else:
                                    if plane_id in free_plane:
                                        free_plane.remove(plane_id)
                                        my_plane.follow_plane = enemy_plane.ID
                                        followed_flag = 1
                                        cmd_list.append(env_cmd.make_followparam(my_plane.ID, enemy_plane.ID, my_plane.para["move_max_speed"], my_plane.para['move_max_acc'], my_plane.para['move_max_g']))
                        
                        if followed_flag == 0:# 回到敌机上一次出现的位置搜查
                            dis = 999999
                            follow_plane = None
                            for fre_ID in free_plane:
                                my_plane = self.get_body_info_by_id(self.my_plane, fre_ID)
                                if my_plane.Type == 1 and self.revenge == False:
                                    continue
                                if my_plane.ready_missile> 0:
                                    tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                                    if tmp_dis < dis and self.is_in_center(my_plane, center_radius=44000)==True:
                                        dis = tmp_dis
                                        follow_plane = my_plane
                            if follow_plane is not None and self.is_in_center(follow_plane, center_radius=35000):
                                free_plane.remove(follow_plane.ID)
                                cmd_list.append(env_cmd.make_linepatrolparam(follow_plane.ID, [enemy_plane.pos3d,], follow_plane.para["move_max_speed"],
                                                follow_plane.para["move_max_acc"], follow_plane.para["move_max_g"]))
                            elif follow_plane is not None:
                                follow_plane.follow_plane = None
            # 扩大视野模块
            if len(self.enemy_leader_plane)==2:
                # 先给我方有人机视野
                for enemy in self.enemy_plane:
                    for leader in self.my_leader_plane:
                        if leader.Availability and enemy.can_see(leader) and len(enemy.followed_plane)==0 and enemy.ready_missile>0:
                            dis = 99999999
                            follow_plane = None
                            for my_plane_id in free_plane:
                                my_plane = self.get_body_info_by_id(self.my_plane, my_plane_id)
                                if TSVector3.distance(my_plane.pos3d, enemy.pos3d) < dis and my_plane.Type == 2:
                                    dis = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                                    follow_plane = my_plane
                            if follow_plane is not None:
                                enemy.followed_plane.append(follow_plane.ID)
                                follow_plane.follow_plane = enemy.ID
                                free_plane.remove(follow_plane.ID)
                                if enemy.lost_flag:
                                    cmd_list.append(env_cmd.make_linepatrolparam(follow_plane.ID, [enemy.pos3d,], follow_plane.para["move_max_speed"],
                                                            follow_plane.para["move_max_acc"], follow_plane.para["move_max_g"]))
                                else:
                                    cmd_list.append(env_cmd.make_followparam(follow_plane.ID, enemy.ID, follow_plane.para["move_max_speed"], follow_plane.para['move_max_acc'], follow_plane.para['move_max_g']))
                # 在跟踪其他飞机
                for my_plane_id in free_plane.copy():
                    my_plane = self.get_body_info_by_id(self.my_plane, my_plane_id)
                    if my_plane.follow_plane == None and my_plane.Type == 2:
                        dis = 999999999
                        for enemy_plane in self.enemy_uav_plane:
                            if enemy_plane.Availability and len(enemy_plane.followed_plane) == 0 and enemy_plane.ready_missile>0:
                                if TSVector3.distance(enemy_plane.pos3d, my_plane.pos3d) < dis:
                                    dis = TSVector3.distance(enemy_plane.pos3d, my_plane.pos3d)
                                    my_plane.follow_plane = enemy_plane.ID
                    if my_plane.follow_plane is not None:
                        enemy_plane = self.get_body_info_by_id(self.enemy_plane, my_plane.follow_plane)
                        enemy_plane.followed_plane.append(my_plane.ID)
                        free_plane.remove(my_plane.ID)
                        if enemy_plane.lost_flag and enemy.Availability:
                            cmd_list.append(env_cmd.make_linepatrolparam(my_plane.ID, [enemy_plane.pos3d,], my_plane.para["move_max_speed"],
                                                    my_plane.para["move_max_acc"], my_plane.para["move_max_g"]))
                        else:
                            cmd_list.append(env_cmd.make_followparam(my_plane.ID, enemy_plane.ID, my_plane.para["move_max_speed"], my_plane.para['move_max_acc'], my_plane.para['move_max_g']))   

            # 躲避模块
            for plane in self.my_plane:
                dis = 9999999
                enemy = None
                for missile_id in plane.locked_missile_list:
                    missile = self.get_body_info_by_id(self.missile_list, missile_id)
                    if TSVector3.distance(plane.pos3d, missile.pos3d) < dis and sim_time - missile.arrive_time<-3:
                        dis = TSVector3.distance(plane.pos3d, missile.pos3d)
                        enemy = missile
                if enemy is None:
                    continue
                if plane.Type == 1:
                    plane.evade_leader(enemy, cmd_list)
                else:
                    plane.evade(enemy, cmd_list)
            # 干扰模块
            if len(need_jam_list):
                self.activate_jam(cmd_list, need_jam_list, sim_time)
            self.init_move(free_plane, cmd_list, sim_time)

    def init_pos(self, sim_time, cmd_list):
        self.formation()
        # 初始化部署
        if sim_time == 2:
            leader_plane_1 = self.my_leader_plane[0]
            leader_plane_2 = self.my_leader_plane[1]
            # 初始化有人机位置
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -135000 * self.side, 50000, 9000, 400, self.init_direction))
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_2.ID, -135000 * self.side, -50000, 9000, 400, self.init_direction))
            for i, plane in enumerate(self.my_uav_plane):
                if i < 6:
                    if i != 5:
                        cmd_list.append(
                            env_cmd.make_entityinitinfo(plane.ID, -125000 * self.side, 150000 - (i+1) * 50000, 9000, 300, self.init_direction))
                    else:
                        cmd_list.append(
                            env_cmd.make_entityinitinfo(plane.ID, -135000 * self.side, 0, 9000, 300, self.init_direction))
                else:
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(plane.ID, -145000 * self.side, 75000 - (i+1)%3 * 50000, 9000, 300, self.init_direction))

    def init_move(self, free_plane, cmd_list, sim_time):
        for plane_ID in free_plane:
            plane = self.get_body_info_by_id(self.my_plane, plane_ID)
            if plane.Type==2 and len(self.enemy_plane):
                if sim_time%2:
                    relative_pitch = math.pi/5
                else:
                    relative_pitch = -math.pi/5 
            else:
                relative_pitch = 0
            if (self.side == 1 and plane.X > self.bound * self.side) or (self.side == -1 and plane.X < self.bound * self.side):
            # if plane.X > self.bound * self.side :
                init_direction = (self.init_direction + 180) % 360
            else:
                if (plane.Heading*180/math.pi - self.init_direction)<5 or sim_time<600:
                    init_direction = self.init_direction
                else:
                    init_direction = (self.init_direction + 180) % 360
            distance = 15*plane.para["move_max_speed"]
            new_dir = TSVector3.calorientation(init_direction/180*math.pi, relative_pitch)
            new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, distance))
            if plane.Type==1:
                new_pos['Z'] = 6500
                if len(self.enemy_uav_plane)>0:
                    for enemy in self.enemy_plane:
                        if enemy.Z-60*math.sin(10/180*math.pi)<new_pos['Z']:
                            new_pos['Z'] = enemy.Z-60*math.sin(10/180*math.pi)-100
                if new_pos['Z'] < 2000:
                    new_pos['Z'] = 12500
            route_list = [{"X": new_pos['X'], "Y": new_pos['Y'], "Z": new_pos['Z']},]
            if init_direction != self.init_direction or len(self.enemy_leader_plane) == 2:
                if (plane.Z<5000 or plane.Z>10000) and plane.Type==2:
                    new_pos['Z'] = 9000  
                route_list = [{"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000},]
            if len(self.enemy_plane) and self.is_in_center(plane, center_radius=40000)==False:
                route_list = [{"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000},]
            # 没有敌人，朝对方中心点移动
            if plane.Type==1 and sim_time>100:
                cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 200, 1, 6))
            elif plane.Type==2:
                cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 300, 2, 12))

    def can_attack_plane(self, enemy_plane, evade_plane_id, free_plane):
        can_attack_dict = {}
        for my_plane in self.first_formation:
            if my_plane.Availability:
                attack_plane = None
                tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                left_weapon = my_plane.ready_missile > 0
                in_range = my_plane.can_attack(enemy_plane)
                if in_range and left_weapon:
                    attack_plane = my_plane
                    enemy_can_see_me = False
                    for enemy in self.enemy_plane:
                        if enemy.can_see(attack_plane):
                            enemy_can_see_me = True
                    if enemy_can_see_me == False:
                        can_attack_dict[attack_plane.ID] = tmp_dis
        can_attack_list = [key for key, value in sorted(can_attack_dict.items(), key=lambda d: d[1])]
        if can_attack_list:
            uav_plane = None
            for plane_id in can_attack_list:
                if self.get_body_info_by_id(self.my_plane, plane_id).Type == 2:
                    uav_plane = plane_id
                    break
            if uav_plane:
                attack_plane = self.get_body_info_by_id(self.my_plane, uav_plane)
            else:
                attack_plane = self.get_body_info_by_id(self.my_plane, can_attack_list[0])
        else:
            attack_plane = None      
        return attack_plane

    def activate_jam(self, cmd_list, need_jam_list,sim_time):
        for comba in self.jam_list:
            if comba[0].Availability:
                if sim_time - comba[2]>=60:
                    comba[1] = 0
                if comba[1] == 0 and need_jam_list:
                    need_jam_flag = False
                    for need_jam in need_jam_list.copy():
                        enemy_missile = self.get_body_info_by_id(self.missile_list, need_jam["missile_entity"])
                        enemy_plane = self.get_body_info_by_id(self.enemy_plane, enemy_missile.LauncherID)
                        my_plane = self.get_body_info_by_id(self.my_plane, enemy_missile.EngageTargetID)
                        remove_flag = 0
                        if enemy_plane.Availability and enemy_plane.can_see(my_plane) and comba[0].can_see(enemy_plane, see_factor=0.95):
                            remove_flag = 1
                        else:
                            for enemy in self.enemy_plane:
                                if enemy.Availability and enemy.can_see(my_plane) and comba[0].can_see(enemy, see_factor=0.95):
                                    remove_flag = 1
                        if remove_flag:
                            need_jam_list.remove(need_jam)
                            need_jam_flag = True
                    if need_jam_flag:
                        comba[1] = 1
                        comba[2] = sim_time + 4
                        cmd_list.append(env_cmd.make_jamparam(comba[0].ID))

    def is_in_center(self, plane, center_radius=50000):
        distance_to_center = (plane.X**2 + plane.Y**2 + (plane.Z - 9000)**2)**0.5
        if distance_to_center <= center_radius and plane.Z >= 2000 and plane.Z <= 15000:
            return True
        return False