import copy
from turtle import distance
from typing import List
import numpy as np
import math
import random
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.agent import Agent
from agent.tink_AI_v2.agent_base import Plane, Missile
#定义 一个 TSVector3D

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

        # 第一编队
        self.first_formation = {}
        # 第二编队
        self.sec_formation = {}

        # 躲避列表
        self.evade_list = []
        # 干扰列表
        self.need_jam_list = []

    def reset(self):
        self._init()

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        cmd_list = []
        self.update_decision(sim_time, obs_side, cmd_list)
        return cmd_list

    def angle(self, a, b):
        if BaseTSVector3.iszero(a) or BaseTSVector3.iszero(b):
            return 0
        else:
            ma = BaseTSVector3.length(a)
            mb = BaseTSVector3.length(b)
            mab = BaseTSVector3.dot(a, b)
            if mab / ma / mb>1:
                tmp_ans = 1
            else:
                tmp_ans = mab / ma / mb
        return math.acos(tmp_ans)

    def update_dead_info(self, my_plane, enemy_plane):
        for plane in my_plane:
            if plane.lost_flag:
                plane.Availability = 0
        for plane in enemy_plane:
            if plane.lost_flag: 
                if plane.close_distance != -1 and len(plane.locked_missile_list)>0:
                    dead_flag = 0
                    for missile_id in plane.locked_missile_list:
                        tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                        if tmp_missile.lost_flag:
                            dead_flag = 1
                            break
                    if dead_flag:
                        plane.Availability = 0
            else:
                plane.Availability = 1
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
            # 更新导弹追踪信息
            for missile_id in plane.locked_missile_list.copy():
                tmp_missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if tmp_missile.lost_flag:
                    plane.locked_missile_list.remove(missile_id)

    def update_entity_info(self, obs_side):
        # 己方所有飞机信息
        self.my_plane = self.get_all_body_list(self.my_plane, obs_side['platforminfos'], Plane)
        # 敌方所有飞机信息
        self.enemy_plane = self.get_all_body_list(self.enemy_plane, obs_side['trackinfos'], Plane)
        # 获取双方导弹信息
        self.missile_list = self.get_all_body_list(self.missile_list, obs_side['missileinfos'], Missile)
        enemy_missile = []
        my_missile = []
        for rocket in self.missile_list:
            if rocket.Identification == self.my_plane[0].Identification:
                if not rocket.lost_flag:
                    my_missile.append(rocket)
            else:
                if not rocket.lost_flag:
                    enemy_missile.append(rocket)
        self.enemy_missile = enemy_missile
        self.my_missile = my_missile
        # 更新阵亡飞机
        self.update_dead_info(self.my_plane, self.enemy_plane)
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
    def get_body_info_by_type(self, agent_info_list, agent_type: int) -> list:
        agent_list = []
        for agent_info in agent_info_list:
            # if agent_type == 3:
            #     avail = 1 - agent_info.lost_flag
            # else:
            #     avail = agent_info.Availability
            if agent_info.Type == agent_type:
                agent_list.append(agent_info)
        return agent_list

    # 通过实体ID获取实体信息
    def get_body_info_by_id(self, agent_info_list, agent_id: int) -> object:
        for agent_info in agent_info_list:
            # if agent_info.Type == 3:
            #     avail = 1 - agent_info.lost_flag
            # else:
            #     avail = agent_info.Availability
            if agent_info.ID == agent_id:
                return agent_info
        return None

    def formation(self):
        self.first_formation["up_plane"] = self.my_uav_plane[0]
        self.first_formation["down_plane"] = self.my_uav_plane[1]
        self.first_formation["leader_plane"] = self.my_leader_plane[0]
        self.first_formation["uav_1"] = self.my_uav_plane[4]
        self.first_formation["uav_2"] = self.my_uav_plane[5]

        self.sec_formation["up_plane"] = self.my_uav_plane[2]
        self.sec_formation["down_plane"] = self.my_uav_plane[3]
        self.sec_formation["leader_plane"] = self.my_leader_plane[1]
        self.sec_formation["uav_1"] = self.my_uav_plane[6]
        self.sec_formation["uav_2"] = self.my_uav_plane[7]
        # 干扰
        self.jam_list = [[plane, 0, 0] for plane in self.my_leader_plane]

    def update_decision(self, sim_time, obs_side, cmd_list):
        self.update_entity_info(obs_side)
        if sim_time <= 2:
            self.init_pos(sim_time, cmd_list)
        else:
            # 开火模块
            # 更新敌人的被打击列表
            undetected_list = self.update_hit_list()
            threat_plane_list = self.get_threat_target_list()
            #更新躲避信息
            need_jam_list = self.update_evade()
            evade_plane_id = [plane["plane_entity"].ID for plane in self.evade_list]
            free_plane = []
            for my_plane in self.my_plane:
                if my_plane.ID not in evade_plane_id and my_plane.Availability:
                    free_plane.append(my_plane)
            threat_plane_list_copy = threat_plane_list.copy()
            attack_enemy = {}
            for threat_plane in threat_plane_list_copy:
                attack_plane = self.can_attack_plane(threat_plane, evade_plane_id, free_plane)
                for enemy in self.hit_enemy_list:
                    if enemy[0].ID == threat_plane.ID:
                        if threat_plane.ID in attack_enemy:
                            attack_enemy[threat_plane.ID] = max(enemy[3], attack_enemy[enemy[0].ID])
                        else:
                            attack_enemy[threat_plane.ID] = enemy[3]
                if attack_plane is not None:
                    can_attack_now = True
                    if threat_plane.ID in attack_enemy.keys() and sim_time - attack_enemy[threat_plane.ID] < 5:
                        can_attack_now = False
                    if can_attack_now:
                        EntityPos = {"X": attack_plane.X, "Y": attack_plane.Y, "Z": attack_plane.Z}
                        EnemyPos = {"X": threat_plane.X, "Y": threat_plane.Y, "Z": threat_plane.Z}
                        distance = TSVector3.distance(EntityPos, EnemyPos)
                        factor_fight = distance/attack_plane.para["launch_range"]
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, factor_fight))
                        self.hit_enemy_list.append([threat_plane, None, None, sim_time, None])
                        threat_plane_list.remove(threat_plane)
            # 制导
            for enemy_plane in self.hit_enemy_list:
                guide_plane = None
                if enemy_plane[2] is not None and enemy_plane[2].can_see(enemy_plane[0]):
                    enemy_plane[4] = enemy_plane[2]
                    for fre_p in free_plane.copy():
                        if fre_p.ID == enemy_plane[2].ID:
                            free_plane.remove(fre_p)
                            guide_plane = enemy_plane[2]
                            break
                    if enemy_plane[0].ID in undetected_list:
                        undetected_list.remove(enemy_plane[0].ID)
                elif enemy_plane[4] is not None and enemy_plane[4].can_see(enemy_plane[0]):
                    if enemy_plane[0].ID in undetected_list:
                        undetected_list.remove(enemy_plane[0].ID)

                if enemy_plane[0].ID in undetected_list:
                    dis = 0
                    if guide_plane is None:
                        for my_plane in free_plane:
                            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane[0].pos3d)
                            if tmp_dis > dis and my_plane.can_see(enemy_plane[0]):
                                guide_plane = my_plane
                                dis = tmp_dis
                        enemy_plane[4] = guide_plane

                    if guide_plane is not None:
                        free_plane.remove(guide_plane)
                        undetected_list.remove(enemy_plane[0].ID)
                        if 150<enemy_plane[0].Speed<300:
                            follow_speed = enemy_plane[0].Speed
                        else:
                            follow_speed = guide_plane.para["move_max_speed"]
                            cmd_list.append(env_cmd.make_followparam(guide_plane.ID, enemy_plane[0].ID, follow_speed, guide_plane.para["move_max_acc"], guide_plane.para["move_max_g"]))

            # 追击模块
            if len(free_plane)>0 and len(threat_plane_list)>0:
                for my_plane in free_plane.copy():
                    dis = 9999999
                    follow_plane = None
                    if my_plane.LeftWeapon > 0:
                        for enemy_plane in threat_plane_list:
                            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                            if enemy_plane.LeftWeapon - len(enemy_plane.used_missile_list) == 0:
                                tmp_dis = tmp_dis*0.1
                            elif my_plane.Type == 1:
                                continue
                            if tmp_dis < dis:
                                follow_plane = enemy_plane
                                dis = tmp_dis
                    if follow_plane is not None:
                        free_plane.remove(my_plane)
                        cmd_list.append(env_cmd.make_followparam(my_plane.ID, follow_plane.ID, my_plane.para["move_max_speed"], my_plane.para['move_max_acc'], my_plane.para['move_max_g']))
                        
            # 躲避模块
            for comba in self.evade_list:
                plane = comba["plane_entity"]
                enemy = comba["missile_entity"]
                if plane.Type == 1:
                    plane.evade_leader(enemy, cmd_list)
                else:
                    plane.evade(enemy, cmd_list)

            # 干扰模块
            if len(need_jam_list):
                self.activate_jam(cmd_list, need_jam_list, sim_time)
            self.init_move(free_plane, cmd_list)
    def init_pos(self, sim_time, cmd_list):
        self.formation()
        # 初始化部署
        if sim_time == 2:
            init_direction = 90
            if self.side == -1:
                init_direction = 270
            leader_plane_1 = self.my_leader_plane[0]
            leader_plane_2 = self.my_leader_plane[1]
            # 初始化有人机位置
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -135000 * self.side, 60000, 9500, 400, init_direction))
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_2.ID, -135000 * self.side, -60000, 9000, 400, init_direction))
            for i, plane in enumerate(self.my_uav_plane):
                if i % 2 ==0:
                     cmd_list.append(
                        env_cmd.make_entityinitinfo(plane.ID, -125000 * self.side, 60000 - i%3 * random.random()*30000, 9000, 300, init_direction))
                else:
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(plane.ID, -125000 * self.side, -60000 + i%3 * random.random()*30000, 9000, 300, init_direction))

    def init_move(self, free_plane, cmd_list):
        for plane in free_plane:
            route = [{"X": 25000 * self.side, "Y": -7500, "Z": 8000}]
            # 没有敌人，朝对方中心点移动
            cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route, 300, 1, 6))

    def _is_dead(self, plane):
        if plane.Availability == 0:
            return True
        else:
            return False

    def update_evade(self):
        missile_list = self.enemy_missile
        evade_id = [comb["missile_entity"].ID for comb in self.evade_list]
        need_jam_list = []
        # 统计所有被导弹瞄准的飞机
        if len(missile_list) != 0:
            for missile in missile_list:
                attacked_plane = self.get_body_info_by_id(self.my_plane, missile.EngageTargetID)
                if attacked_plane.Availability == 0:
                    continue
                if missile.ID not in evade_id:
                    evade_comb = {}
                    evade_comb["plane_entity"] = attacked_plane
                    evade_comb["missile_entity"] = missile
                    evade_comb["distance"] = TSVector3.distance(missile.pos3d,attacked_plane.pos3d)
                    self.evade_list.append(evade_comb)

            # 给危险程度分类
        # 将不需要躲避的移除列表
        evade_list_copy = self.evade_list.copy()
        for attacked_plane in evade_list_copy:
            missile = attacked_plane["missile_entity"]
            plane = attacked_plane["plane_entity"]
            distance = attacked_plane["distance"]
            missile_gone, over_target = False, False
            # 导弹已爆炸
            if self.get_body_info_by_id(self.missile_list, missile.ID).lost_flag:
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
            # 需要干扰
            if 20000< distance < 23000:
                need_jam_list.append({"missile_entity":missile,"distance":distance, "attacked_plane":plane})
            if any([missile_gone, over_target]):
                self.evade_list.remove(attacked_plane)
        need_jam_list = sorted(need_jam_list, key=lambda d: d["distance"], reverse=False)
        return need_jam_list

    def get_threat_target_list(self):
        # 有人机最重要，距离，带弹数量
        threat_dict = {}
        for enemy in self.enemy_plane:
            dis = 99999999
            for my_plane in self.my_plane:
                if my_plane.Availability:
                    dis_tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                    if dis_tmp < dis:
                        dis = dis_tmp
            if enemy.Type == 1:
                # 敌机在距离我方有人机在距离的前提下会多20000的威胁值，并且敌人是有人机会再多10000威胁值
                dis -= 10000
            dis -= 20000
            if dis < 0:
                dis = 0
            if dis not in threat_dict:
                threat_dict[dis] = enemy
            else:
                threat_dict[dis + 0.1] = enemy

        threat_plane_list = [value for key, value in sorted(threat_dict.items(), key=lambda d: d[0])]
        # # 去除有人机
        # threat_plane_list_copy = threat_plane_list.copy()
        # for threat_plane in threat_plane_list_copy:
        #     if threat_plane.Type == 1:
        #         threat_plane_list.remove(threat_plane)

        for hit_enemy in self.hit_enemy_list:
            leader_hit = False
            # 敌有人机可以打两发
            if len(hit_enemy[0].locked_missile_list) < 2 and hit_enemy[0].Type == 2:
                leader_hit = True
            elif len(hit_enemy[0].locked_missile_list) < 3 and hit_enemy[0].Type == 1:
                leader_hit = True
            threat_plane_list_copy = threat_plane_list.copy()
            for threat_plane in threat_plane_list_copy:
                if hit_enemy[0].ID == threat_plane.ID and not leader_hit:
                    threat_plane_list.remove(threat_plane)
        return threat_plane_list

    def can_attack_plane(self, enemy_plane,evade_plane_id,free_plane):
        dis = 9999999
        can_attack_dict = {}
        for my_plane in self.my_plane:
            if my_plane.Availability:
                attack_plane = None
                tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                left_weapon = my_plane.LeftWeapon > 0
                in_range = my_plane.can_attack(enemy_plane)
                is_dead = self._is_dead(my_plane)
                if (in_range and left_weapon) and not is_dead:
                    attack_plane = my_plane
                    if attack_plane.ID not in can_attack_dict:
                        can_attack_dict[attack_plane.ID] = tmp_dis
                if attack_plane is not None and attack_plane.ID in evade_plane_id:# 打出去的导弹要有空闲飞机能制导
                    guide_plane = None
                    for plane in free_plane:
                        if plane.can_see(enemy_plane):
                            guide_plane = plane
                    missile_dis = TSVector3.distance(attack_plane.pos3d, enemy_plane.pos3d)
                    if guide_plane is None and missile_dis>20000:
                        can_attack_dict.pop(attack_plane.ID)
        can_attack_list = [key for key, value in sorted(can_attack_dict.items(), key=lambda d: d[1])]
        if can_attack_list:
            attack_plane = self.get_body_info_by_id(self.my_plane, can_attack_list[0])
        else:
            attack_plane = None      
        return attack_plane

    def update_hit_list(self):
        for comba in self.hit_enemy_list:
            if comba[1] is None:
                for my_missile in self.my_missile:
                    if my_missile.EngageTargetID == comba[0].ID:
                        comba[1] = my_missile
                        comba[2] = self.get_body_info_by_id(self.my_plane, my_missile.LauncherID)
                        if my_missile.ID not in comba[0].locked_missile_list:
                            comba[0].locked_missile_list.append(my_missile.ID)

        undetected_list = []
        undetected_id_list = []
        hit_enemy_list_copy = self.hit_enemy_list.copy()
        for comba in hit_enemy_list_copy:
            is_dead = False
            if self.get_body_info_by_id(self.enemy_plane, comba[0].ID).Availability == 0:
                is_dead = True
            missile_gone = True
            if comba[1] is not None:
                if self.get_body_info_by_id(self.missile_list, comba[1].ID).lost_flag == 0:
                    missile_gone = False
            if is_dead or missile_gone:
                self.hit_enemy_list.remove(comba)
            elif not is_dead and not missile_gone:
                if TSVector3.distance(comba[0].pos3d, comba[1].pos3d) > 20000*0.9:
                    if comba[0].ID not in undetected_id_list:
                        undetected_id_list.append(comba[0].ID)
                        undetected_list.append(comba[0])
        undetected_list = [comba.ID for comba in sorted(undetected_list, key=lambda d: d.Type, reverse=False)]
        return undetected_list

    def activate_jam(self, cmd_list, need_jam_list,sim_time):
        for comba in self.jam_list:
            if comba[0].Availability:
                if sim_time - comba[2]>=60:
                    comba[1] = 0
                if comba[1] == 0 and need_jam_list:
                    need_jam_list_copy = need_jam_list.copy()
                    need_jam_flag = False
                    for need_jam in need_jam_list_copy:
                        enemy_plane = self.get_body_info_by_id(self.enemy_plane, need_jam["missile_entity"].LauncherID)
                        my_plane = self.get_body_info_by_id(self.my_plane, need_jam["missile_entity"].EngageTargetID)
                        remove_flag = 0
                        if enemy_plane.Availability and enemy_plane.can_see(my_plane) and comba[0].can_see(enemy_plane):
                            remove_flag = 1
                        else:
                            for enemy in self.enemy_plane:
                                if enemy.Availability and enemy.can_see(my_plane) and comba[0].can_see(enemy):
                                    remove_flag = 1
                        if remove_flag:
                            need_jam_list.remove(need_jam)
                            need_jam_flag = True
                    if need_jam_flag:
                        comba[1] = 1
                        comba[2] = sim_time+2
                        cmd_list.append(env_cmd.make_jamparam(comba[0].ID))