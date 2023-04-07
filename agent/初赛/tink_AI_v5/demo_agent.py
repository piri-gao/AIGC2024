import copy
from typing import List
import numpy as np
import math
import random
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.agent import Agent
from agent.tink_AI_v5.agent_base import Plane, Missile

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
        # 当前时间
        self.sim_time = 0

    def reset(self):
        self._init()

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        cmd_list = []
        self.update_decision(sim_time, obs_side, cmd_list)
        return cmd_list

    # 更新当前各个实体的存亡以及部分额外记录信息
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
                if (tmp_missile.lost_flag and sim_time - tmp_missile.arrive_time > 3) or tmp_missile.loss_target:
                    plane.locked_missile_list.remove(tmp_missile.ID)
            plane.ready_missile = plane.LeftWeapon - len(plane.used_missile_list)

        for plane in self.enemy_plane:
            if plane.lost_flag>6: 
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
                if missile.loss_target==False and missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list and missile.lost_flag==0:
                    plane.locked_missile_list.append(missile.ID)
            # 更新导弹追踪信息
            for missile_id in plane.locked_missile_list.copy():
                tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if tmp_missile.lost_flag:
                    plane.locked_missile_list.remove(tmp_missile.ID)
            plane.ready_missile = plane.LeftWeapon - len(plane.used_missile_list)

        # 己方有人机信息
        self.my_leader_plane = self.get_body_info_by_type(self.my_plane, 1)
        # 己方无人机信息
        self.my_uav_plane = self.get_body_info_by_type(self.my_plane, 2)
        # 敌方有人机信息
        self.enemy_leader_plane = self.get_body_info_by_type(self.enemy_plane, 1)
        # 敌方无人机信息
        self.enemy_uav_plane = self.get_body_info_by_type(self.enemy_plane, 2)

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

    # 局部视野更新各实体信息
    def update_entity_info(self, obs_side, sim_time):
        # 己方所有飞机信息
        self.my_plane = self.get_all_body_list(self.my_plane, obs_side['platforminfos'], Plane)
        # 敌方所有飞机信息
        self.enemy_plane = self.get_all_body_list(self.enemy_plane, obs_side['trackinfos'], Plane)
        # 获取双方导弹信息
        self.missile_list = self.get_all_body_list(self.missile_list, obs_side['missileinfos'], Missile)
        # 更新阵亡飞机
        self.update_dead_info(sim_time)
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
        
    # 更新真实智能体信息
    def get_all_body_list(self, agent_info_list, obs_list, cls):
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

    # 更新决策
    def update_decision(self, sim_time, obs_side, cmd_list):
        self.update_entity_info(obs_side,sim_time)
        if sim_time <= 2:
            self.init_pos(sim_time, cmd_list)
        else:
            # 更新空闲飞机以及威胁飞机信息模块
            free_plane = []
            no_missile_plane = []
            threat_plane_list = []
            for my_plane in self.my_plane:
                if len(my_plane.locked_missile_list)==0 and my_plane.Availability:
                    if my_plane.Type==1 and my_plane.do_jam:# 该有人机在执行干扰任务
                        continue
                    free_plane.append(my_plane)
                    if my_plane.ready_missile == 0 and my_plane.Type == 2:
                        no_missile_plane.append(my_plane.ID)
            free_plane = sorted(free_plane, key=lambda d: d.ready_missile, reverse=False)
            free_plane = [plane.ID for plane in sorted(free_plane, key=lambda d: d.Type, reverse=True)]
            for enemy in self.enemy_plane:
                if enemy.ready_missile>0 and enemy.Type==2:
                    threat_plane_list.append(enemy.ID)
            # 指定僚机
            if len(self.enemy_plane):
                for leader_plane in self.my_leader_plane:
                    if leader_plane.Availability:
                        if leader_plane.wing_plane != None:
                            wing_plane = self.get_body_info_by_id(self.my_plane, leader_plane.wing_plane)
                            if wing_plane.Availability and len(leader_plane.locked_missile_list)>0:
                                continue
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

            # # 开火模块         
            # attack_enemy = {}
            # for threat_plane_id in threat_plane_list:
            #     threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_plane_id)
            #     if threat_plane.Availability:
            #         if threat_plane.Type==2 and len(threat_plane.locked_missile_list)>2:
            #             continue
            #         threat_ID = threat_plane.ID
            #         attack_plane = self.can_attack_plane(threat_plane)
            #         for missile_id in threat_plane.locked_missile_list:
            #             missile = self.get_body_info_by_id(self.missile_list, missile_id)
            #             if threat_ID in attack_enemy:
            #                 attack_enemy[threat_ID] = max(missile.fire_time, attack_enemy[threat_ID])
            #             else:
            #                 attack_enemy[threat_ID] = missile.fire_time
            #         if attack_plane is not None:
            #             can_attack_now = True
            #             if threat_ID in attack_enemy.keys() and sim_time - attack_enemy[threat_ID] < 7:
            #                 can_attack_now = False
            #             if can_attack_now:
            #                 factor_fight = 1
            #                 cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_ID, factor_fight))
            #                 attack_plane.ready_missile -= 1
            
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
                        # total_pitch = TSVector3.calpitch(total_dir)
                        
                        move_speed = leader.para["move_max_speed"]
                        if min_dis>35000:
                            if leader.Speed<=leader.para["move_max_speed"]:
                                move_speed = leader.Speed
                            turn_ratio = 1/2
                            if self.is_in_center(leader,130000)==False:
                                turn_ratio = 0
                                total_pitch = 0
                            else:
                                total_pitch = math.pi/18
                            if abs(leader.pi_bound(leader.Heading - total_heading - math.pi*turn_ratio))>math.pi*turn_ratio:
                                total_heading = total_heading - math.pi*turn_ratio
                            else:
                                total_heading = total_heading + math.pi*turn_ratio
                            
                        else:
                            if sim_time%2:
                                total_pitch = math.pi/6
                            else:
                                total_pitch = -math.pi/6

                        if total_pitch>0:
                            distance, total_pitch = leader.limit_height(total_pitch, leader.para["area_max_alt"])
                        else:
                            distance, total_pitch = leader.limit_height(total_pitch, leader.Z-2000)

                        total_dir = TSVector3.calorientation(total_heading, total_pitch)
                        total_dir = TSVector3.normalize(total_dir)
                        # distance = 15 * move_speed
                        evade_pos = TSVector3.plus(leader.pos3d, TSVector3.multscalar(total_dir, distance))
                        straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": evade_pos['Z']}, ]

                        cmd_list.append(env_cmd.make_linepatrolparam(leader.ID, straight_evade_route_list,
                                            move_speed, leader.para["move_max_acc"], leader.para["move_max_g"]))
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
                            follow_missile = None
                            dis = 9999999
                            for missile_id in leader.locked_missile_list:
                                missile = self.get_body_info_by_id(self.missile_list, missile_id)
                                if TSVector3.distance(leader.pos3d, missile.pos3d) < dis and sim_time - missile.arrive_time<2:
                                    dis = TSVector3.distance(leader.pos3d, missile.pos3d)
                                    follow_missile = missile
                            if follow_missile != None:
                                cmd_list.append(env_cmd.make_followparam(wing_plane.ID, follow_missile.ID, wing_plane.para["move_max_speed"], wing_plane.para['move_max_acc'], wing_plane.para['move_max_g']))
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


            # 无人机躲避模块
            for plane in self.my_uav_plane:
                dis = 9999999
                enemy = None
                for missile_id in plane.locked_missile_list:
                    missile = self.get_body_info_by_id(self.missile_list, missile_id)
                    if TSVector3.distance(plane.pos3d, missile.pos3d) < dis and sim_time - missile.arrive_time<2:
                        dis = TSVector3.distance(plane.pos3d, missile.pos3d)
                        enemy = missile
                if enemy is None:
                    continue
                if plane.Type==2:
                    plane.evade(enemy, cmd_list)
                else:
                    plane.evade_leader(enemy,cmd_list)

            # 干扰模块
            self.activate_jam(cmd_list, sim_time)
            self.init_move(free_plane, cmd_list, sim_time)

    #初始化我方飞机位置
    def init_pos(self, sim_time, cmd_list):
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

    # 我方飞机无威胁且空闲情况下飞机路径
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
                new_pos['Z'] = 13000
                # if len(self.enemy_uav_plane)>0:
                #     for enemy in self.enemy_plane:
                #         if enemy.Z-60*math.sin(10/180*math.pi)<new_pos['Z']:
                #             new_pos['Z'] = enemy.Z-60*math.sin(10/180*math.pi)-100
                # if new_pos['Z'] < 2000:
                #     new_pos['Z'] = 12500
            route_list = [{"X": new_pos['X'], "Y": new_pos['Y'], "Z": new_pos['Z']},]
            if init_direction != self.init_direction or len(self.enemy_leader_plane) == 2:
                if (plane.Z<5000 or plane.Z>10000) and plane.Type==2:
                    new_pos['Z'] = 9000  
                route_list = [{"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000},]
            if len(self.enemy_plane) and self.is_in_center(plane, center_radius=40000)==False:
                route_list = [{"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": 9000},]
            # 没有敌人，朝对方中心点移动
            if plane.Type==1 and sim_time>0:
                cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 200, 1, 6))
            elif plane.Type==2:
                cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 300, 2, 12))

    # 找到可以攻击敌方飞机的己方飞机
    def can_attack_plane(self, enemy_plane):
        can_attack_dict = {}
        for my_plane in self.first_formation:
            if my_plane.Availability:
                attack_plane = None
                tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                left_weapon = my_plane.ready_missile > 0
                attack_dis = 0
                if enemy_plane.Type==2:
                    attack_dis = 15000
                in_range = my_plane.can_attack(enemy_plane,attack_dis)
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

    # 实施干扰
    def activate_jam(self, cmd_list, sim_time):
        need_jam_list = []
        evade_list = []
        for i, plane in enumerate(self.my_leader_plane):
            if plane.Availability==0:
                continue
            dis = 9999999
            enemy = None
            for missile_id in plane.locked_missile_list:
                missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if self.my_leader_plane[1-i].Availability:
                    leader_turn_time = max(self.get_turn_time(self.my_leader_plane[i], missile),self.get_turn_time(self.my_leader_plane[1-i], missile))
                else:
                    leader_turn_time = self.get_turn_time(self.my_leader_plane[i], missile)
                if (leader_turn_time+6)*1000 + 20000 < TSVector3.distance(plane.pos3d, missile.pos3d) < (leader_turn_time+8)*1000 + 20000 and plane.ID not in need_jam_list:
                    need_jam_list.append(plane.ID)
                if TSVector3.distance(plane.pos3d, missile.pos3d) < dis and sim_time - missile.arrive_time<3:
                    dis = TSVector3.distance(plane.pos3d, missile.pos3d)
                    enemy = missile
            if enemy is not None:
                plane.close_missile = enemy
                evade_list.append(plane.ID)
                plane.evade_leader(enemy, cmd_list)
        leader1 = self.my_leader_plane[0]
        leader2 = self.my_leader_plane[1]
        # 找到最中间的敌机进行干扰
        middle_enemy_dict = {}
        for need_jam_id in need_jam_list:
            middle_enemy_pos3d = {"X": 0, "Y": 0, "Z": 0}
            leader = self.get_body_info_by_id(self.my_plane, need_jam_id)
            for enemy in self.enemy_plane:
                if enemy.lost_flag==0 and enemy.can_see(leader):
                    dis = TSVector3.distance(leader.pos3d, enemy.pos3d)
                    if enemy.ID not in middle_enemy_dict:
                        middle_enemy_dict[enemy.ID] = dis
            middle_enemy_list = [key for key, value in sorted(middle_enemy_dict.items(), key=lambda d: d[1])]
            if len(middle_enemy_list):
                leader.middle_enemy_plane = middle_enemy_list[int(len(middle_enemy_list)/2)]
            else:
                leader.middle_enemy_plane = None
        if len(need_jam_list):
            if leader1.ID not in need_jam_list and leader1.Availability:# 1号有人机空闲，2号有人机需要干扰支持
                if sim_time-leader1.last_jam>60 and sim_time - leader2.jammed>=6:
                    if self.plane_can_jam(leader2, leader1, evade_list,sim_time ,cmd_list) == False:
                        self.find_way_jam(leader1, leader2.middle_enemy_plane, sim_time, cmd_list)
                elif 6<sim_time - leader1.last_jam:
                    leader1.do_jam = False
                if leader2.do_jam==False:
                    leader2.evade_leader(leader2.close_missile, cmd_list)

            elif leader2.ID not in need_jam_list and leader2.Availability:# 2号有人机空闲，1号有人机需要干扰支持
                if sim_time-leader1.last_jam>60 and sim_time - leader1.jammed>=6:
                    if self.plane_can_jam(leader1, leader2,  evade_list, sim_time, cmd_list) == False:
                        self.find_way_jam(leader2, leader1.middle_enemy_plane, sim_time, cmd_list)
                elif 6<=sim_time - leader2.last_jam:
                    leader2.do_jam = False
                if leader1.do_jam == False:
                    leader1.evade_leader(leader1.close_missile, cmd_list)
            else:# 1号、2号有人机需要干扰支持，各自顾自己
                self.save_myself(leader1, need_jam_list, sim_time, cmd_list)
                self.save_myself(leader2, need_jam_list, sim_time, cmd_list) 
        else:
            leader1.do_jam = False
            leader1.evade_leader(leader1.close_missile, cmd_list)
            leader2.do_jam = False
            leader2.evade_leader(leader2.close_missile, cmd_list)
    
    # 有人机的干扰自顾自
    def save_myself(self, leader, need_jam_list, sim_time, cmd_list):
        if leader.ID in need_jam_list and leader.do_jam==False:
            enemy = self.get_body_info_by_id(self.enemy_plane, leader.middle_enemy_plane)
            can_jam = False
            if sim_time-leader.last_jam>60  and sim_time - leader.jammed>=6 and enemy is not None:
                acc_time = (leader.para["move_max_speed"] - leader.Speed)/9.8 + 12 
                if TSVector3.distance(leader.pos3d, leader.close_missile.pos3d)/leader.close_missile.Speed>acc_time and leader.can_see(enemy):
                    leader.do_jam = True
                    cmd_list.append(env_cmd.make_jamparam(leader.ID))
                    leader.jammed = sim_time+1
                    leader.last_jam = sim_time+6
                    can_jam = True
                elif TSVector3.distance(leader.pos3d, leader.close_missile.pos3d)/leader.close_missile.Speed > acc_time + self.get_turn_time(leader, enemy):
                    self.find_way_jam(leader, leader.middle_enemy_plane, sim_time, cmd_list)
                    can_jam = True
                else:
                    can_jam = False
            if can_jam == False:
                leader.do_jam = False
                leader.evade_leader(leader.close_missile, cmd_list)

    # 飞机拐弯时间
    def get_turn_time(self,leader, enemy):
        my_turn_theta = abs(
            leader.pi_bound(leader.XY2theta(enemy.X - leader.X, enemy.Y - leader.Y) - leader.Heading))
        look_theta = math.pi*leader.para['radar_heading']/2/180
        if my_turn_theta > look_theta:
            my_turn_theta -= look_theta
        else:
            my_turn_theta = 0
        my_turn_t = my_turn_theta / (leader.para['move_max_g'] * 9.8) * leader.Speed
        return my_turn_t

    # 判断有人机是否可以进行干扰
    def plane_can_jam(self, need_jam_plane, jam_plane, evade_list, sim_time, cmd_list):
        if need_jam_plane.middle_enemy_plane is not None:
            enemy = self.get_body_info_by_id(self.enemy_plane, need_jam_plane.middle_enemy_plane)
            if enemy.lost_flag==0:
                can_jam = False
                if not need_jam_plane.can_see(enemy, see_factor=0.6,jam_dis=117600) and jam_plane.can_see(enemy, see_factor=0.6,jam_dis=117600):
                    if jam_plane.ID not in evade_list:
                        cmd_list.append(env_cmd.make_followparam(jam_plane.ID, enemy.ID, jam_plane.Speed, jam_plane.para['move_max_acc'], jam_plane.para['move_max_g']))
                        can_jam = True
                    elif jam_plane.can_see(enemy, see_factor=0.6, jam_dis=117600):
                        can_jam = True
                    else:
                        can_jam = False
                elif need_jam_plane.can_see(enemy, see_factor=0.6,jam_dis=117600) and sim_time-need_jam_plane.last_jam>60 and sim_time - need_jam_plane.jammed>=6 and TSVector3.distance(need_jam_plane.close_missile.pos3d, need_jam_plane.pos3d)>15000:
                    need_jam_plane.do_jam = True
                    cmd_list.append(env_cmd.make_jamparam(need_jam_plane.ID))
                    need_jam_plane.jammed = sim_time+1
                    need_jam_plane.last_jam = sim_time+6
                    return True
                if can_jam:
                    jam_plane.do_jam = True
                    cmd_list.append(env_cmd.make_jamparam(jam_plane.ID))
                    need_jam_plane.jammed = sim_time+1
                    jam_plane.last_jam = sim_time+6
                    return True
        return False
    
    # 跟踪可以进行干扰但不在探测范围的敌机
    def find_way_jam(self, jam_plane, middle_enemy_id, sim_time, cmd_list):
        if middle_enemy_id==None:
            return
        jam_plane.do_jam = True
        jam_enemy = self.get_body_info_by_id(self.enemy_plane, middle_enemy_id)
        if jam_enemy.lost_flag:
            jam_enemy = None
        if jam_enemy is not None:
            cmd_list.append(env_cmd.make_followparam(jam_plane.ID, jam_enemy.ID, jam_plane.Speed, jam_plane.para['move_max_acc'], jam_plane.para['move_max_g']))
                
    # 判断是否飞机实体在中心
    def is_in_center(self, plane, center_radius=50000):
        distance_to_center = (plane.X**2 + plane.Y**2 + (plane.Z - 9000)**2)**0.5
        if distance_to_center <= center_radius and plane.Z >= 2000 and plane.Z <= 15000:
            return True
        return False