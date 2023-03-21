import copy
import numpy as np
import math
import random
from typing import List

from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.agent import Agent
from agent.QYKZ_Aggressive.agent_base import Plane, Missile

class AIAgent(Agent):

    def __init__(self, name, config):
        super(AIAgent, self).__init__(name, config['side'])
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
        self.dead_enemy_list = []
        # 敌方无人机信息列表
        self.enemy_uav_plane = []

        # 我方导弹信息
        self.my_missile = []
        self.runing_my_missile_list = []
        # 敌方导弹信息
        self.enemy_missile = []
        self.runing_enemy_missile_list = []
        self.missile_list = []
        # 被打击的敌机列表
        self.hit_enemy_list = []

        # 被制导信息
        self.guide_plane_dict = {}

        # 识别红军还是蓝军
        self.side = None

        # 编队
        self.formation = []

        # 躲避列表
        self.evade_list = []

        self.stage = 0
        self.leader_dead_list = []

    def reset(self):
        self._init()

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        cmd_list = []
        self.update_decision(sim_time, obs_side, cmd_list)
        return cmd_list

    def update_entity_info(self, obs_side, sim_time):
        # 己方所有飞机信息
        self.my_plane = self.get_all_entity(self.my_plane, obs_side['platforminfos'], Plane)
        # 敌方所有飞机信息
        self.enemy_plane = self.get_all_entity(self.enemy_plane, obs_side['trackinfos'], Plane)
        # 获取双方导弹信息
        self.missile_list = self.get_all_entity(self.missile_list, obs_side['missileinfos'], Missile)
        enemy_missile = []
        my_missile = []
        for rocket in self.missile_list:
            if rocket.Identification == self.my_plane[0].Identification:
                if rocket.lost_flag == 0:
                    my_missile.append(rocket)
            else:
                if rocket.lost_flag == 0:
                    enemy_missile.append(rocket)
        self.enemy_missile = enemy_missile
        self.my_missile = my_missile
        # 更新阵亡飞机
        self.update_dead_info(sim_time)
        # 己方有人机信息
        self.my_leader_plane = self.get_entity_info_by_type(self.my_plane, 1)
        # 己方无人机信息
        self.my_uav_plane = self.get_entity_info_by_type(self.my_plane, 2)
        # 敌方有人机信息
        self.enemy_leader_plane = self.get_entity_info_by_type(self.enemy_plane, 1)
        # 敌方无人机信息
        self.enemy_uav_plane = self.get_entity_info_by_type(self.enemy_plane, 2)
        self.calc_score()
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
    
    def update_dead_info(self, sim_time):
        for plane in self.enemy_plane:
            if plane.lost_flag and plane.close_distance != -1 and len(plane.locked_missile_list) > 0:
                is_dead = False
                for missile_id in plane.locked_missile_list:
                    missle_item = self.get_entity_info_by_id(self.missile_list, missile_id)
                    if missle_item.lost_flag:
                        is_dead = True
                        break
                if is_dead:
                    plane.Availability = 0
                    if plane.Type == 1 and plane.ID not in self.leader_dead_list:
                        self.leader_dead_list.append(plane.ID)
                        self.stage += 1
            else:
                plane.Availability = 1
                if self.in_center(plane):
                    plane.center_time += 1
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                if missile.lost_flag == 0 and missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list:
                    plane.locked_missile_list.append(missile.ID)
            for missile_id in plane.locked_missile_list.copy():
                missle_item = self.get_entity_info_by_id(self.missile_list, missile_id)
                if missle_item.lost_flag:
                    plane.locked_missile_list.remove(missle_item.ID)

        for plane in self.my_plane:
            if plane.lost_flag:
                plane.Availability = 0
            elif self.in_center(plane):
                plane.center_time += 1
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                if missile.lost_flag == 0 and missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list:
                    plane.locked_missile_list.append(missile.ID)
            minDis = 99999999
            plane.closest_missile = None
            for missile_id in plane.locked_missile_list:
                missile = self.get_entity_info_by_id(self.missile_list, missile_id)
                if TSVector3.distance(plane.pos3d, missile.pos3d) < minDis:
                    minDis = TSVector3.distance(plane.pos3d, missile.pos3d)
                    plane.closest_missile = missile
            plane.ready_missile = plane.LeftWeapon - len(plane.used_missile_list)

        if self.revenge == False:
            self.formation = self.my_uav_plane
            enemy_flag = False
            enemy_left_weapon = 0
            for plane in self.enemy_plane:
                if plane.Availability:
                    enemy_flag = True
                    enemy_left_weapon += plane.LeftWeapon - len(plane.used_missile_list)
            if enemy_flag and enemy_left_weapon < 5 or sim_time > 20 * 60 - 300:
                self.revenge = True
        else:
            self.formation = self.my_plane

    def get_all_entity(self, agent_info_list, obs_list, cls):
        # 更新真实智能体信息
        for obs_agent_info in obs_list:
            agent_in_list = False
            for agent_info in agent_info_list:
                if obs_agent_info['ID'] == agent_info.ID:
                    agent_info.update_agent_info(obs_agent_info)
                    agent_info.lost_flag = 0
                    agent_in_list = True
                    break
            if not agent_in_list:
                agent_info_list.append(cls(obs_agent_info))

        # 更改死亡的智能体状态
        for agent_info in agent_info_list:
            agent_in_list = False
            for obs_agent_info in obs_list:
                if obs_agent_info['ID'] == agent_info.ID:
                    agent_in_list = True
                    break
            if not agent_in_list:
                agent_info.lost_flag = 1
        return agent_info_list

    # 通过一种类型获取实体信息
    def get_entity_info_by_type(self, agent_info_list, agent_type: int) -> list:
        agent_list = []
        for agent_info in agent_info_list:
            if agent_info.Type == agent_type:
                agent_list.append(agent_info)
        return agent_list

    # 通过实体ID获取实体信息
    def get_entity_info_by_id(self, agent_info_list, agent_id: int) -> object:
        for agent_info in agent_info_list:
            if agent_info.ID == agent_id:
                return agent_info
        return None

    def update_decision(self, sim_time, obs_side, cmd_list):
        self.update_entity_info(obs_side,sim_time)
        if sim_time <= 2:
            self.init_pos(sim_time, cmd_list)
        else:
            # 更新信息
            self.update_evade()                 
            evade_plane_id = [plane["plane_entity"] for plane in self.evade_list]
            free_plane = self.get_free_list(evade_plane_id)
            undetected_list = self.get_undetected_list()
            threat_plane_list = self.get_threat_list()
            follow_plane_list = threat_plane_list.copy()
            
            # 干扰模块
            self.jam(sim_time, cmd_list)
            # 开火模块
            self.attack(free_plane, evade_plane_id, threat_plane_list, cmd_list, sim_time)
            # 制导模块
            self.guidance(free_plane, undetected_list, cmd_list)
            # 追击模块
            self.chase(free_plane, follow_plane_list, cmd_list)
            # 躲避模块
            self.evade(cmd_list)
            # 其余飞机执行默认指令
            self.init_move(free_plane, cmd_list, sim_time)

    def attack(self, free_plane, evade_plane_id, threat_plane_list, cmd_list, sim_time):
        attack_enemy = {}
        for threat_ID in threat_plane_list.copy():
            threat_plane = self.get_entity_info_by_id(self.enemy_plane, threat_ID)
            attack_plane = self.can_attack(threat_plane, evade_plane_id, free_plane)
            for enemy in self.hit_enemy_list:
                if enemy[0] == threat_ID:
                    if threat_ID in attack_enemy:
                        attack_enemy[threat_ID] = max(enemy[3], attack_enemy[enemy[0]])
                    else:
                        attack_enemy[threat_ID] = enemy[3]
            if attack_plane is not None:
                attack_now = True
                cd_time = 0 if self.my_score < self.enemy_score + 36 else 3
                if threat_ID in attack_enemy.keys() and sim_time - attack_enemy[threat_ID] < cd_time:
                    attack_now = False
                if sim_time > 60 * 19:
                    if self.my_score - self.enemy_score == 1 and self.my_center_time < self.enemy_center_time:
                        attack_now = False
                    elif self.my_score == self.enemy_score and self.my_center_time > self.enemy_center_time + (1200 - sim_time) * (2 - self.stage):
                        attack_now = False

                if attack_now:
                    MyPos = {"X": attack_plane.X, "Y": attack_plane.Y, "Z": attack_plane.Z}
                    EnemyPos = {"X": threat_plane.X, "Y": threat_plane.Y, "Z": threat_plane.Z}
                    distance = TSVector3.distance(MyPos, EnemyPos)
                    factor = (distance + distance / 900 * threat_plane.para["move_max_speed"]) / attack_plane.para["launch_range"]
                    cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_ID, factor))
                    attack_plane.ready_missile -= 1
                    self.my_score -= 1
                    self.hit_enemy_list.append([threat_plane.ID, None, None, sim_time, None])
                    if self.enemy_strategy_defense():
                        self.revenge = True
                    
                    threat_plane_list.remove(threat_ID)

    def guidance(self, free_plane, undetected_list, cmd_list):
        for hit_enemy in self.hit_enemy_list:
            enemy_plane = self.get_entity_info_by_id(self.enemy_plane, hit_enemy[0])
            guide_plane = None
            if enemy_plane.ID not in undetected_list:
                continue
            old_guide_plane = None
            tmp_guide_plane = None
            if hit_enemy[4] is not None:
                old_guide_plane = self.get_entity_info_by_id(self.my_plane, hit_enemy[4])
            if hit_enemy[2] is not None:
                tmp_guide_plane = self.get_entity_info_by_id(self.my_plane, hit_enemy[2])

            if old_guide_plane is not None and old_guide_plane.visible(enemy_plane) and old_guide_plane.ID in free_plane:
                if enemy_plane.ID in undetected_list:
                    guide_plane = old_guide_plane
            elif tmp_guide_plane is not None and tmp_guide_plane.visible(enemy_plane):
                if tmp_guide_plane.ID in free_plane:
                    guide_plane = tmp_guide_plane
                    hit_enemy[4] = tmp_guide_plane.ID

            dis = 9999999
            if guide_plane is None:
                for free_plane_ID in free_plane:
                    my_plane = self.get_entity_info_by_id(self.my_plane, free_plane_ID)
                    tmp = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                    if tmp < dis and my_plane.visible(enemy_plane):
                        dis = tmp
                        guide_plane = my_plane
                if guide_plane is not None:
                    hit_enemy[4] = guide_plane.ID

            if guide_plane is not None:
                free_plane.remove(guide_plane.ID)
                undetected_list.remove(enemy_plane.ID)
                if 150 < enemy_plane.Speed < 300:
                    follow_speed = enemy_plane.Speed
                else:
                    follow_speed = guide_plane.para["move_max_speed"]
                    cmd_list.append(env_cmd.make_followparam(guide_plane.ID, enemy_plane.ID, follow_speed, guide_plane.para["move_max_acc"], guide_plane.para["move_max_g"]))

    def chase(self, free_plane, follow_plane_list, cmd_list):
        for enemy_plane_ID in follow_plane_list:
            enemy_plane = self.get_entity_info_by_id(self.enemy_plane, enemy_plane_ID)
            if len(enemy_plane.followed_plane) > 0:
                for followed_plane_id in enemy_plane.followed_plane.copy():
                    if followed_plane_id in free_plane:
                        my_plane = self.get_entity_info_by_id(self.my_plane, followed_plane_id)
                        if my_plane.Availability == 0:
                            enemy_plane.followed_plane.remove(followed_plane_id)
                        else:
                            free_plane.remove(followed_plane_id)
                    else:
                        enemy_plane.followed_plane.remove(followed_plane_id) 

        if len(free_plane) > 0 and len(follow_plane_list) > 0:
            for enemy_plane_ID in follow_plane_list:
                enemy_plane = self.get_entity_info_by_id(self.enemy_plane, enemy_plane_ID)
                if enemy_plane.Type == 1 and len(enemy_plane.followed_plane) < 4 + 4 * self.stage:
                    dis = 9999999
                    follow_plane = None
                    for free_plane_ID in free_plane:
                        my_plane = self.get_entity_info_by_id(self.my_plane, free_plane_ID)
                        if my_plane.Type == 1 and self.revenge == False:
                            continue
                        if my_plane.ready_missile > 0:
                            tmp = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                            if enemy_plane.Type == 1:
                                tmp = tmp * 0.1
                            elif enemy_plane.LeftWeapon - len(enemy_plane.used_missile_list) > 0:
                                tmp = tmp * 0.5
                            if tmp < dis:
                                dis = tmp
                                follow_plane = my_plane
                    if follow_plane is not None:
                        enemy_plane.followed_plane.append(follow_plane.ID)
                        free_plane.remove(follow_plane.ID)
                for plane_id in enemy_plane.followed_plane:
                    my_plane = self.get_entity_info_by_id(self.my_plane, plane_id)
                    cmd_list.append(env_cmd.make_followparam(my_plane.ID, enemy_plane_ID, my_plane.para["move_max_speed"], my_plane.para['move_max_acc'], my_plane.para['move_max_g']))

    def evade(self, cmd_list):
        evade_flag = []
        for evadeinfo in self.evade_list:
            plane = self.get_entity_info_by_id(self.my_plane, evadeinfo["plane_entity"])
            if plane.ID in evade_flag:
                continue
            evade_flag.append(plane.ID)
            enemy = self.get_entity_info_by_id(self.missile_list, evadeinfo["missile_entity"])
            plane.evade(enemy, cmd_list)

    def jam(self, sim_time, cmd_list):
        jam_list = []
        evade_list = []
        for i, plane in enumerate(self.my_leader_plane):
            if plane.Availability == 0:
                continue
            if sim_time - plane.last_jam > 6:
                plane.in_jam = False
            if plane.closest_missile is not None:
                evade_list.append(plane.ID)

            leader_1 = self.my_leader_plane[i]
            leader_2 = self.my_leader_plane[1-i]
            for missile_id in plane.locked_missile_list:
                missile = self.get_entity_info_by_id(self.missile_list, missile_id)
                if leader_2.Availability and sim_time - leader_2.last_jam > 60:
                    if sim_time - self.my_leader_plane[i].last_jam < 60:
                        turn_time = self.get_turn_time(leader_2, missile)
                    else:
                        turn_time = min(self.get_turn_time(leader_1, missile), self.get_turn_time(leader_2, missile))
                elif sim_time - leader_1.last_jam > 60:
                    turn_time = self.get_turn_time(leader_1, missile)
                else:
                    turn_time = 99999999
                if (turn_time + 6) * 1000 + 20000 < TSVector3.distance(plane.pos3d, missile.pos3d) < (turn_time + 8) * 1000 + 20000 and plane.ID not in jam_list:
                    jam_list.append(plane.ID)

            enemy_dict = {}
            for jam_id in jam_list:
                enemy_pos3d = {"X": 0, "Y": 0, "Z": 0}
                leader = self.get_entity_info_by_id(self.my_plane, jam_id)
                for enemy in self.enemy_plane:
                    if enemy.lost_flag == 0 and enemy.visible(leader):
                        dis = leader.bound_in_pi(TSVector3.calheading(TSVector3.minus(enemy.pos3d, leader.pos3d)) - leader.Heading)
                        if enemy.ID not in enemy_dict:
                            enemy_dict[enemy.ID] = dis
                enemy_list = [k for k,v in sorted(enemy_dict.items(), key=lambda d:d[1])]
                if len(enemy_list) > 0:
                    leader.jam_plane = enemy_list[int(len(enemy_list) / 2)]
                else:
                    leader.jam_plane = None

            leader_1 = self.my_leader_plane[0]
            leader_2 = self.my_leader_plane[1]
            if len(jam_list) > 0:
                if leader_1.ID not in jam_list and leader_1.Availability:
                    save_flag = False
                    if sim_time - leader_1.last_jam > 60 and sim_time - leader_2.jammed >= 6:
                        if self.can_jam(sim_time, leader_2, leader_1, evade_list, cmd_list) == False:
                            if self.check_jam(leader_2, leader_1, leader_2.jam_plane):
                                self.find_jam(leader_1, leader_2.jam_plane, cmd_list)
                                save_flag = True
                        else:
                            save_flag = True
                    if save_flag == False:
                        self.self_jam(sim_time, leader_2, jam_list, cmd_list)
                elif leader_2.ID not in jam_list and leader_2.Availability:
                    save_flag = False
                    if sim_time - leader_2.last_jam > 60 and sim_time - leader_1.jammed >= 6:
                        if self.can_jam(sim_time, leader_1, leader_2, evade_list, cmd_list) == False:
                            if self.check_jam(leader_1, leader_2, leader_1.jam_plane):
                                self.find_jam(leader_2, leader_1.jam_plane, cmd_list)
                                save_flag = True
                        else:
                            save_flag = True
                    if save_flag == False:
                        self.self_jam(sim_time, leader_1, jam_list, cmd_list)
                else:
                    self.self_jam(sim_time, leader_1, jam_list, cmd_list)
                    self.self_jam(sim_time, leader_2, jam_list, cmd_list)

    def get_turn_time(self, plane, missile):
        turn_theta = abs(plane.bound_in_pi(plane.XY2theta(missile.X-plane.X, missile.Y-plane.Y) - plane.Heading))
        radar_theta = plane.para['radar_heading'] * math.pi / 360
        if turn_theta > radar_theta:
            turn_theta -= radar_theta
        else:
            turn_theta = 0
        return turn_theta / (plane.para['move_max_g'] * 9.8) * plane.Speed

    def can_jam(self, sim_time, need_plane, help_plane, evade_list, cmd_list):
        if need_plane.jam_plane is not None:
            enemy = self.get_entity_info_by_id(self.enemy_plane, need_plane.jam_plane)
            if enemy.lost_flag == 0:
                can_jam = False
                if not need_plane.visible(enemy) and help_plane.visible(enemy):
                    if help_plane.ID not in evade_list:
                        cmd_list.append(env_cmd.make_followparam(help_plane.ID, enemy.ID, help_plane.Speed, help_plane.para['move_max_acc'], help_plane.para['move_max_g']))
                        can_jam = True
                    elif help_plane.visible(enemy):
                        can_jam = True
                elif need_plane.visible(enemy) and sim_time - need_plane.last_jam > 60 and sim_time - need_plane.jammed >= 6 and TSVector3.distance(need_plane.closest_missile.pos3d, need_plane.pos3d) > 15000:
                    need_plane.in_jam = True
                    cmd_list.append(env_cmd.make_jamparam(need_plane.ID))
                    need_plane.jammed = sim_time+1
                    need_plane.last_jam = sim_time+7
                    need_plane.jam_enemy = None
                    return True
                if can_jam:
                    help_plane.in_jam = True
                    cmd_list.append(env_cmd.make_jamparam(help_plane.ID))
                    help_plane.jammed = sim_time+1
                    help_plane.last_jam = sim_time+7
                    help_plane.jam_enemy = None
                    return True
        return False

    def check_jam(self, need_plane, help_plane, enemy_id, checkdis=20000):
        if enemy_id == None:
            return
        jam_target = self.get_entity_info_by_id(self.enemy_plane, enemy_id)
        turn_time = self.get_turn_time(help_plane, jam_target)
        for missile_id in need_plane.locked_missile_list:
            missile = self.get_entity_info_by_id(self.missile_list, missile_id)
            if TSVector3.distance(missile.pos3d, need_plane.pos3d) > checkdis + missile.Speed * (turn_time + 6):
                return True
        return False

    def find_jam(self, plane, enemy_id, cmd_list):
        if enemy_id == None:
            return
        plane.in_jam = True
        jam_target = self.get_entity_info_by_id(self.enemy_plane, enemy_id)
        if jam_target.lost_flag == 0:
            cmd_list.append(env_cmd.make_followparam(plane.ID, jam_target.ID, plane.Speed, plane.para['move_max_acc'], plane.para['move_max_g']))
        else:
            cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, [jam_target.pos3d,], plane.para["move_max_speed"], plane.para["move_max_acc"], plane.para["move_max_g"]))

    def self_jam(self, sim_time, plane, jam_list, cmd_list):
        if plane.ID in jam_list and plane.in_jam == False:
            enemy = self.get_entity_info_by_id(self.enemy_plane, plane.jam_plane)
            can_jam = False
            if sim_time - plane.last_jam > 60 and sim_time - plane.jammed >= 6 and enemy is not None:
                acceleration = (plane.para['move_max_speed'] - plane.Speed) / 9.8 + 12
                if TSVector3.distance(plane.pos3d, plane.closest_missile.pos3d) / 1000 > acceleration and plane.visible(enemy):
                    plane.in_jam = True
                    cmd_list.append(env_cmd.make_jamparam(plane.ID))
                    plane.jammed = sim_time + 1
                    plane.last_jam = sim_time + 7
                    can_jam = True
                elif TSVector3.distance(plane.pos3d, plane.closest_missile.pos3d)/1000 > acceleration + self.get_turn_time(plane, enemy):
                    self.find_jam(plane, plane.jam_plane, cmd_list)
                    can_jam = True
            if can_jam == False:
                plane.in_jam = False
                return False
            return True

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

    def init_move(self, free_plane, cmd_list, sim_time):
        for plane_ID in free_plane:
            plane = self.get_entity_info_by_id(self.my_plane, plane_ID)
            if plane.Type == 2:
                if sim_time % 2:
                    relative_pitch = math.pi / 5
                else:
                    relative_pitch = -math.pi / 5 
            else:
                relative_pitch = 0
            if (self.side == 1 and plane.X > self.bound * self.side) or (self.side == -1 and plane.X < self.bound * self.side):
                init_direction = (self.init_direction + 180) % 360
            else:
                if (plane.Heading * 180/math.pi - self.init_direction) < 5 or sim_time < 600:
                    init_direction = self.init_direction
                else:
                    init_direction = (self.init_direction + 180) % 360
            distance = 20 * plane.para["move_max_speed"]
            new_dir = TSVector3.calorientation(init_direction / 180 * math.pi, relative_pitch)
            new_pos = TSVector3.plus(plane.pos3d, TSVector3.multscalar(new_dir, distance))
            if plane.Type == 1:
                new_pos['Z'] = 9000
                if len(self.enemy_uav_plane) > 0:
                    for enemy in self.enemy_plane:
                        if enemy.Z - 60 * math.sin(10 / 180 * math.pi) < new_pos['Z']:
                            new_pos['Z'] = enemy.Z - 60 * math.sin(10 / 180 * math.pi) - 100
                if new_pos['Z'] < 2000:
                    new_pos['Z'] = 14000
            route_list = [{"X": new_pos['X'], "Y": new_pos['Y'], "Z": new_pos['Z']},]
            if init_direction != self.init_direction:
                route_list = [{"X": (random.random() * 2 - 1) * 3000, "Y": (random.random() * 2 - 1) * 3000, "Z": plane.Z},]

            # 没有敌人，朝对方中心点移动
            if plane.Type == 1 and sim_time > 240:
                cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 400, 1, 6))
            elif plane.Type == 2:
                cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 300, 2, 12))

    def update_evade(self):
        missile_list = self.enemy_missile
        evade_id = [comb["missile_entity"] for comb in self.evade_list]

        # 统计所有被导弹瞄准的飞机
        if len(missile_list) != 0:
            for missile in missile_list:
                attacked_plane = self.get_entity_info_by_id(self.my_plane, missile.EngageTargetID)
                if attacked_plane.Availability == 0:
                    continue
                if missile.ID not in evade_id:
                    evade_comb = {}
                    evade_comb["plane_entity"] = attacked_plane.ID
                    evade_comb["missile_entity"] = missile.ID
                    evade_comb["distance"] = TSVector3.distance(missile.pos3d,attacked_plane.pos3d)
                    self.evade_list.append(evade_comb)

            # 给危险程度分类
        # 将不需要躲避的移除列表
        for attacked_plane in self.evade_list.copy():
            missile = self.get_entity_info_by_id(self.missile_list, attacked_plane["missile_entity"])
            plane = self.get_entity_info_by_id(self.my_plane, attacked_plane["plane_entity"])
            distance = attacked_plane["distance"]
            missile_gone, over_target = False, False
            # 导弹已爆炸
            if self.get_entity_info_by_id(self.missile_list, missile.ID).lost_flag:
                missile_gone = True
            # 过靶
            missile_vector_3d = TSVector3.calorientation(missile.Heading, missile.Pitch)
            missile_vector = np.array([missile_vector_3d["X"], missile_vector_3d["Y"]])
            missile_mp_vector_3d = TSVector3.minus(plane.pos3d, missile.pos3d)
            missile_mp_vector = np.array([missile_mp_vector_3d["X"], missile_mp_vector_3d["Y"]])
            res = np.dot(np.array(missile_vector), np.array(missile_mp_vector)) / (
                    np.sqrt(np.sum(np.square(np.array(missile_vector)))) + 1e-9) / (
                              np.sqrt(np.sum(np.square(np.array(missile_mp_vector)))) + 1e-9)
            if abs(res) > 1:
                res = res / abs(res)
            dir = math.acos(res) * 180 / math.pi
            if abs(dir) > 90:
                over_target = True
            if any([missile_gone, over_target]):
                self.evade_list.remove(attacked_plane)

    def get_undetected_list(self):
        for hitinfo in self.hit_enemy_list:
            if hitinfo[1] is None:
                for my_missile in self.my_missile:
                    if my_missile.EngageTargetID == hitinfo[0]:
                        hitinfo[1] = my_missile.ID
                        hitinfo[2] = my_missile.LauncherID

        undetected_plane = []
        undetected_id = []
        for hitinfo in self.hit_enemy_list.copy():
            enemy_plane = self.get_entity_info_by_id(self.enemy_plane, hitinfo[0])
            my_missile = self.get_entity_info_by_id(self.missile_list, hitinfo[1]) if hitinfo[1] is not None else None
            missile_gone = my_missile is None or my_missile.lost_flag != 0
            if enemy_plane.Availability == 0 or missile_gone:
                self.hit_enemy_list.remove(hitinfo)
            elif enemy_plane.Availability != 0 and not missile_gone:
                if TSVector3.distance(enemy_plane.pos3d, my_missile.pos3d) > 20000*0.8:
                    if enemy_plane.ID not in undetected_id:
                        undetected_id.append(enemy_plane.ID)
                        undetected_plane.append(enemy_plane)
        undetected_plane = [hitinfo.ID for hitinfo in sorted(undetected_plane, key=lambda d: d.Type, reverse=False)]
        return undetected_plane

    def get_threat_list(self):
        # 有人机最重要，距离，带弹数量
        threat_dict = {}
        for enemy in self.enemy_plane:
            if enemy.Availability == 0:
                continue
            dis = 99999999
            for my_plane in self.my_plane:
                if my_plane.Availability:
                    tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                    if tmp < dis:
                        dis = tmp
            enemy.close_distance = dis

            if self.stage < 2 and enemy.Type != 1:      # 在所有有人机被击毁前只打有人机
                continue
            threat_point = dis - 20000
            if enemy.Type == 1:
                threat_point -= 10000
            threat_point = max(threat_point, 0)
            if threat_point not in threat_dict:
                threat_dict[threat_point] = enemy.ID
            else:
                threat_dict[threat_point + 0.1] = enemy.ID
        threat_plane_list = [value for key, value in sorted(threat_dict.items(), key=lambda d: d[0])]
        
        if self.stage < 2:              # 在所有有人机被击毁前只打有人机
            return threat_plane_list
        
        # 去除有人机
        for threat_ID in threat_plane_list.copy():
            threat_plane = self.get_entity_info_by_id(self.enemy_plane, threat_ID)
            if threat_plane.Type == 2:
                threat_plane_list.remove(threat_ID)
        for threat_ID in threat_plane_list.copy():
            if threat_ID not in threat_plane_list:
                continue
            threat_plane = self.get_entity_info_by_id(self.enemy_plane, threat_ID)
            if threat_plane.Type == 2 and threat_plane.LeftWeapon - len(threat_plane.used_missile_list) == 0:
                threat_plane_list.remove(threat_ID) # 去掉不带导弹的敌机无人机
                continue
            for hitinfo in self.hit_enemy_list:
                enemy = self.get_entity_info_by_id(self.enemy_plane, hitinfo[0])
                leader_hit = (enemy.Type == 1) or (enemy.Type == 2 and len(enemy.locked_missile_list) < 3)
                for threat_ID in threat_plane_list.copy():
                    if enemy.ID == threat_ID and not leader_hit:
                        threat_plane_list.remove(threat_ID)
        return threat_plane_list

    def get_free_list(self, evade_plane_id):
        free_plane = []
        for my_plane in self.my_plane:
            if my_plane.Type == 1 and my_plane.in_jam:
                continue
            if my_plane.ID not in evade_plane_id and my_plane.Availability:
                free_plane.append(my_plane)
        free_plane = sorted(free_plane, key=lambda d: d.ready_missile, reverse=False)
        free_plane = [plane.ID for plane in sorted(free_plane, key=lambda d: d.Type, reverse=True)]
        return free_plane

    def can_attack(self, enemy_plane,evade_plane_id,free_plane):
        can_attack_dict = {}
        for my_plane in self.formation:
            if my_plane.Availability:
                attack_plane = None
                dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                if my_plane.attackable(enemy_plane) and my_plane.ready_missile > 0:
                    attack_plane = my_plane
                    if attack_plane.ID not in can_attack_dict:
                        can_attack_dict[attack_plane.ID] = dis
                if attack_plane is not None and attack_plane.ID in evade_plane_id:
                    guide_plane = None
                    for plane_ID in free_plane:
                        plane = self.get_entity_info_by_id(self.my_plane, plane_ID)
                        if plane.visible(enemy_plane):
                            guide_plane = plane
                    missile_dis = TSVector3.distance(attack_plane.pos3d, enemy_plane.pos3d)
                    visible_to_enemy = False
                    for enemy in self.enemy_plane:
                        if enemy.visible(attack_plane):
                            visible_to_enemy = True
                    if (guide_plane is None or visible_to_enemy) and missile_dis > 15000:
                        can_attack_dict.pop(attack_plane.ID)
        can_attack_list = [key for key, value in sorted(can_attack_dict.items(), key=lambda d: d[1])]
        if can_attack_list:
            uav_plane = None
            for plane_id in can_attack_list:
                if self.get_entity_info_by_id(self.my_plane, plane_id).Type == 2:
                    uav_plane = plane_id
                    break
            if uav_plane:
                attack_plane = self.get_entity_info_by_id(self.my_plane, uav_plane)
            else:
                attack_plane = self.get_entity_info_by_id(self.my_plane, can_attack_list[0])
        else:
            attack_plane = None      
        return attack_plane

    def enemy_strategy_defense(self):
        defence = True
        for plane in self.enemy_plane:
            if len(plane.used_missile_list) != 0:
                defence = False
        return defence

    def in_center(self, plane):
        distance_to_center = (plane.X**2 + plane.Y**2 + (plane.Z - 9000)**2)**0.5
        if distance_to_center <= 50000 and plane.Z >= 2000 and plane.Z <= 16000:
            return True
        return False

    def calc_score(self):
        self.my_score = 0
        self.enemy_score = 0
        self.my_center_time = 0
        self.enemy_center_time = 0
        for plane in self.my_plane:
            if plane.Availability == 1:
                if plane.Type == 1:
                    self.my_center_time += plane.center_time
                self.my_score += plane.LeftWeapon - len(plane.used_missile_list)
                if plane.Type == 1:
                    self.my_score += 60
                else:
                    self.my_score += 5
        for plane in self.enemy_plane:
            if plane.Availability == 1:
                if plane.Type == 1:
                    self.enemy_center_time += plane.center_time
                self.enemy_score += plane.LeftWeapon - len(plane.used_missile_list)
                if plane.Type == 1:
                    self.enemy_score += 60
                else:
                    self.enemy_score += 5

        if len(self.enemy_plane) != 10:
            self.enemy_score += (2 - len(self.enemy_leader_plane)) * 64
            self.enemy_score += (8 - len(self.enemy_uav_plane)) * 7
        return self.enemy_score < self.my_score

class BaseTSVector3:
    # 初始化
    def __init__(self,x:float,y:float,z:float):
        return {"X": x, "Y":y, "Z": z}

    # 矢量a 点乘 矢量b
    @staticmethod
    def dot(a, b):
        return a["X"] * b["X"] + a["Y"] * b["Y"] + a["Z"] * b["Z"]

    # 判断矢量a是否为0矢量
    @staticmethod
    def iszero(a):
        if a["X"] == 0 and a["Y"] == 0 and a["Z"] == 0:
            return True
        else:
            return False

    # 计算矢量a的长度
    @staticmethod
    def length(a):
        if a["X"] == 0 and a["Y"] == 0 and a["Z"] == 0:
            return 0
        return math.sqrt(a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"])