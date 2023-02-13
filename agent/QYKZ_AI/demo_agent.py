import copy
from turtle import distance
from typing import List
import numpy as np
import math
import random
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.agent import Agent
from agent.QYKZ_AI.agent_base import Plane, Missile

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

        # 第一编队
        self.first_formation = []
        # 第二编队
        self.sec_formation = []

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

    def update_dead_info(self, sim_time):
        for plane in self.my_plane:
            if plane.lost_flag:
                plane.Availability = 0
            elif self.in_center(plane):
                plane.center_time += 1
            for missile in self.missile_list:
                if missile.LauncherID == plane.ID and missile.ID not in plane.used_missile_list:
                    plane.used_missile_list.append(missile.ID)
                if missile.EngageTargetID == plane.ID and missile.ID not in plane.locked_missile_list and missile.lost_flag==0:
                    plane.locked_missile_list.append(missile.ID)
            plane.ready_missile = plane.LeftWeapon -len(plane.used_missile_list)
        for plane in self.enemy_plane:
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
                if self.in_center(plane):
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
        # 更新是否反击
        if self.revenge == False:
            self.first_formation = self.my_uav_plane
            enemy_left_weapon = 0
            have_enemy = 0
            for plane in self.enemy_plane:
                if plane.Availability:
                    have_enemy = 1
                    enemy_left_weapon += plane.LeftWeapon -len(plane.used_missile_list)
            if have_enemy and enemy_left_weapon < 5 or sim_time > 20 * 60 -300:
                self.revenge = True
        else:
            self.first_formation = self.my_plane

    def update_entity_info(self, obs_side, sim_time):
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

    def update_decision(self, sim_time, obs_side, cmd_list):
        self.update_entity_info(obs_side,sim_time)
        if sim_time <= 2:
            self.init_pos(sim_time, cmd_list)
        else:
            # 更新信息
            need_jam_list = self.update_evade()                 
            evade_plane_id = [plane["plane_entity"] for plane in self.evade_list]
            free_plane = self.get_free_list(evade_plane_id)
            undetected_list = self.get_undetected_list()
            threat_plane_list = self.get_threat_list()
            
            # 开火模块
            self.attack(free_plane, evade_plane_id, threat_plane_list, cmd_list, sim_time)
            # 制导模块
            self.guidance(free_plane, undetected_list, cmd_list)
            # 追击模块
            self.chase(free_plane, threat_plane_list, cmd_list)
            # 躲避模块
            self.evade(cmd_list)
            # 其余飞机执行默认指令
            self.init_move(free_plane, cmd_list, sim_time)

    def attack(self, free_plane, evade_plane_id, threat_plane_list, cmd_list, sim_time):
        attack_enemy = {}
        for threat_ID in threat_plane_list.copy():
            threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_ID)
            attack_plane = self.can_attack_plane(threat_plane, evade_plane_id, free_plane)
            for enemy in self.hit_enemy_list:
                if enemy[0] == threat_ID:
                    if threat_ID in attack_enemy:
                        attack_enemy[threat_ID] = max(enemy[3], attack_enemy[enemy[0]])
                    else:
                        attack_enemy[threat_ID] = enemy[3]
            if attack_plane is not None:
                can_attack_now = True
                if threat_ID in attack_enemy.keys() and sim_time - attack_enemy[threat_ID] < 3:
                    can_attack_now = False
                if can_attack_now:
                    EntityPos = {"X": attack_plane.X, "Y": attack_plane.Y, "Z": attack_plane.Z}
                    EnemyPos = {"X": threat_plane.X, "Y": threat_plane.Y, "Z": threat_plane.Z}
                    distance = TSVector3.distance(EntityPos, EnemyPos)
                    enemy_turn_t = self.get_enemy_turn_time(attack_plane, threat_plane)
                    factor_fight = (distance + distance/900*threat_plane.para["move_max_speed"])/attack_plane.para["launch_range"]
                    cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_ID, factor_fight))
                    attack_plane.ready_missile -= 1
                    self.hit_enemy_list.append([threat_plane.ID, None, None, sim_time, None])
                    if self.enemy_strategy():
                        self.revenge = True
                    threat_plane_list.remove(threat_ID)

    def guidance(self, free_plane, undetected_list, cmd_list):
        for hit_enemy in self.hit_enemy_list:
            enemy_plane = self.get_body_info_by_id(self.enemy_plane, hit_enemy[0])
            guide_plane = None
            if enemy_plane.ID not in undetected_list:
                continue
            tmp_guide_plane = None
            old_guide_plane = None
            if hit_enemy[4] is not None:
                old_guide_plane = self.get_body_info_by_id(self.my_plane, hit_enemy[4])
            if hit_enemy[2] is not None:
                tmp_guide_plane = self.get_body_info_by_id(self.my_plane, hit_enemy[2])
            if tmp_guide_plane is not None and tmp_guide_plane.visible(enemy_plane):
                if tmp_guide_plane.ID in free_plane:
                    hit_enemy[4] = tmp_guide_plane.ID
                    guide_plane = tmp_guide_plane
            elif old_guide_plane and old_guide_plane.visible(enemy_plane) and old_guide_plane.ID in free_plane:
                if enemy_plane.ID in undetected_list:
                    guide_plane = old_guide_plane

            dis = 9999999
            if guide_plane is None:
                for fre_ID in free_plane:
                    my_plane = self.get_body_info_by_id(self.my_plane, fre_ID)
                    tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                    if tmp_dis < dis and my_plane.visible(enemy_plane):
                        guide_plane = my_plane
                        dis = tmp_dis
                if guide_plane is not None:
                    hit_enemy[4] = guide_plane.ID

            if guide_plane is not None:
                free_plane.remove(guide_plane.ID)
                undetected_list.remove(enemy_plane.ID)
                if 150<enemy_plane.Speed<300:
                    follow_speed = enemy_plane.Speed
                else:
                    follow_speed = guide_plane.para["move_max_speed"]
                    cmd_list.append(env_cmd.make_followparam(guide_plane.ID, enemy_plane.ID, follow_speed, guide_plane.para["move_max_acc"], guide_plane.para["move_max_g"]))

    def chase(self, free_plane, threat_plane_list, cmd_list):
        for enemy_plane_ID in threat_plane_list:
            enemy_plane = self.get_body_info_by_id(self.enemy_plane, enemy_plane_ID)
            if len(enemy_plane.followed_plane)>0:
                for followed_plane_id in enemy_plane.followed_plane.copy():
                    if followed_plane_id in free_plane:
                        my_plane = self.get_body_info_by_id(self.my_plane, followed_plane_id)
                        if my_plane.Availability==0:
                            enemy_plane.followed_plane.remove(followed_plane_id)
                        else:
                            free_plane.remove(followed_plane_id)
                    else:
                        enemy_plane.followed_plane.remove(followed_plane_id) 

        if len(free_plane)>0 and len(threat_plane_list)>0:
            for enemy_plane_ID in threat_plane_list:
                enemy_plane = self.get_body_info_by_id(self.enemy_plane, enemy_plane_ID)
                if len(enemy_plane.followed_plane)==0 or (len(enemy_plane.followed_plane)<4 and enemy_plane.Type==1):
                    dis = 9999999
                    follow_plane = None
                    for fre_ID in free_plane:
                        my_plane = self.get_body_info_by_id(self.my_plane, fre_ID)
                        if my_plane.Type == 1 and self.revenge == False:
                            continue
                        if my_plane.ready_missile> 0:
                            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                            if enemy_plane.Type==1:
                                tmp_dis = tmp_dis*0.1
                            elif enemy_plane.LeftWeapon - len(enemy_plane.used_missile_list) > 0:
                                tmp_dis = tmp_dis*0.5
                            if tmp_dis < dis:
                                follow_plane = my_plane
                                dis = tmp_dis
                    if follow_plane is not None:
                        enemy_plane.followed_plane.append(follow_plane.ID)
                        free_plane.remove(follow_plane.ID)
                for plane_id in enemy_plane.followed_plane:
                    my_plane = self.get_body_info_by_id(self.my_plane, plane_id)
                    cmd_list.append(env_cmd.make_followparam(my_plane.ID, enemy_plane_ID, my_plane.para["move_max_speed"], my_plane.para['move_max_acc'], my_plane.para['move_max_g']))

    def evade(self, cmd_list):
        evade_flag = []
        for comba in self.evade_list:
            plane = self.get_body_info_by_id(self.my_plane, comba["plane_entity"])
            if plane.ID in evade_flag:
                continue
            evade_flag.append(plane.ID)
            enemy = self.get_body_info_by_id(self.missile_list, comba["missile_entity"])
            plane.evade(enemy, cmd_list)

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
                    # cmd_list.append(
                    #     env_cmd.make_entityinitinfo(plane.ID, -145000 * self.side, 150000 - (i+1)%3 * 100000, 9000, 300, self.init_direction))
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(plane.ID, -145000 * self.side, 75000 - (i+1)%3 * 50000, 9000, 300, self.init_direction))

    def init_move(self, free_plane, cmd_list, sim_time):
        for plane_ID in free_plane:
            plane = self.get_body_info_by_id(self.my_plane, plane_ID)
            if plane.Type==2:
                if sim_time%2:
                    relative_pitch = math.pi/5
                else:
                    relative_pitch = -math.pi/5 
            else:
                relative_pitch = 0
            if plane.X > self.bound * self.side :
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
            if init_direction != self.init_direction:
                route_list = [{"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": plane.Z},]
            # 没有敌人，朝对方中心点移动
            if plane.Type==1 and sim_time>240:
                cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 400, 1, 6))
            elif plane.Type==2:
                cmd_list.append(env_cmd.make_linepatrolparam(plane.ID, route_list, 300, 2, 12))

    def _is_dead(self, plane):
        if plane.Availability == 0:
            return True
        else:
            return False

    def update_evade(self):
        missile_list = self.enemy_missile
        evade_id = [comb["missile_entity"] for comb in self.evade_list]
        need_jam_list = []
        # 统计所有被导弹瞄准的飞机
        if len(missile_list) != 0:
            for missile in missile_list:
                attacked_plane = self.get_body_info_by_id(self.my_plane, missile.EngageTargetID)
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
            missile = self.get_body_info_by_id(self.missile_list, attacked_plane["missile_entity"])
            plane = self.get_body_info_by_id(self.my_plane, attacked_plane["plane_entity"])
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
            if 20000< distance < 23000 and plane.Type==1:
                need_jam_list.append({"missile_entity":missile.ID,"distance":distance, "attacked_plane":plane.ID})
            if any([missile_gone, over_target]):
                self.evade_list.remove(attacked_plane)
        need_jam_list = sorted(need_jam_list, key=lambda d: d["distance"], reverse=False)
        return need_jam_list

    def win_now(self):
        my_score = 0
        enemy_score = 0
        my_center_time = 0
        enemy_center_time = 0
        for plane in self.my_plane:
            my_center_time += plane.center_time
            if plane.Availability == 1:
                my_score += plane.LeftWeapon - len(plane.used_missile_list)
                if plane.Type == 2:
                    my_score += 5
                else:
                    my_score += 60
        for plane in self.enemy_plane:
            enemy_center_time += plane.center_time
            if plane.Availability == 1:
                enemy_score += plane.LeftWeapon - len(plane.used_missile_list)
                if plane.Type == 2:
                    enemy_score += 5
                else:
                    enemy_score += 60
        if enemy_score == my_score:
            enemy_score += enemy_center_time
            my_score += my_center_time
        if enemy_score<my_score:
            return True
        else:
            return False

    def enemy_strategy(self):
        defence = True
        for plane in self.enemy_plane:
            if len(plane.used_missile_list) != 0:
                defence = False
        return defence

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
            enemy_plane = self.get_body_info_by_id(self.enemy_plane, hitinfo[0])
            my_missile = self.get_body_info_by_id(self.missile_list, hitinfo[1]) if hitinfo[1] is not None else None
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

            threat_point = dis - 20000    # 敌机在距离我方有人机在距离的前提下会多20000的威胁值，并且敌人是有人机会再多10000威胁值
            if enemy.Type == 1:
                threat_point -= 10000
            threat_point = max(threat_point, 0)
            if threat_point not in threat_dict:
                threat_dict[threat_point] = enemy.ID
            else:
                threat_dict[threat_point + 0.1] = enemy.ID
        threat_plane_list = [value for key, value in sorted(threat_dict.items(), key=lambda d: d[0])]
        
        
        # 去除有人机
        for threat_ID in threat_plane_list.copy():
            threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_ID)
            if threat_plane.Type == 2:
                threat_plane_list.remove(threat_ID)
        for threat_ID in threat_plane_list.copy():
            if threat_ID not in threat_plane_list:
                continue
            threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_ID)
            if threat_plane.Type == 2 and threat_plane.LeftWeapon - len(threat_plane.used_missile_list) == 0:
                threat_plane_list.remove(threat_ID) # 去掉不带导弹的敌机无人机
                continue
            for hitinfo in self.hit_enemy_list:
                enemy = self.get_body_info_by_id(self.enemy_plane, hitinfo[0])
                leader_hit = (enemy.Type == 1) or (enemy.Type == 2 and len(enemy.locked_missile_list) < 3)
                for threat_ID in threat_plane_list.copy():
                    if enemy.ID == threat_ID and not leader_hit:
                        threat_plane_list.remove(threat_ID)
        return threat_plane_list

    def get_free_list(self, evade_plane_id):
        free_plane = []
        for my_plane in self.my_plane:
            if my_plane.ID not in evade_plane_id and my_plane.Availability:
                free_plane.append(my_plane)
        free_plane = sorted(free_plane, key=lambda d: d.ready_missile, reverse=False)
        free_plane = [plane.ID for plane in sorted(free_plane, key=lambda d: d.Type, reverse=True)]
        return free_plane

    def can_attack_plane(self, enemy_plane,evade_plane_id,free_plane):
        can_attack_dict = {}
        for my_plane in self.first_formation:
            if my_plane.Availability:
                attack_plane = None
                tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                left_weapon = my_plane.ready_missile > 0
                in_range = my_plane.attackable(enemy_plane)
                if in_range and left_weapon:
                    attack_plane = my_plane
                    if attack_plane.ID not in can_attack_dict:
                        can_attack_dict[attack_plane.ID] = tmp_dis
                if attack_plane is not None and attack_plane.ID in evade_plane_id:# 打出去的导弹要有空闲飞机能制导
                    guide_plane = None
                    for plane_ID in free_plane:
                        plane = self.get_body_info_by_id(self.my_plane, plane_ID)
                        if plane.visible(enemy_plane):
                            guide_plane = plane
                    missile_dis = TSVector3.distance(attack_plane.pos3d, enemy_plane.pos3d)
                    # 对方没看到我以及我离对方15000时可以发起攻击
                    enemy_can_see_me = False
                    for enemy in self.enemy_plane:
                        if enemy.visible(attack_plane):
                            enemy_can_see_me = True
                    if (guide_plane is None or enemy_can_see_me) and missile_dis > 15000:
                        can_attack_dict.pop(attack_plane.ID)
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
    
    def get_enemy_turn_time(self, my_plane, enemy):
        turn_theta = enemy.Heading - my_plane.Heading
        while turn_theta > math.pi:
            turn_theta -= 2*math.pi
        while turn_theta <= -math.pi:
            turn_theta += 2*math.pi
        turn_theta = abs(turn_theta)
        turn_t = turn_theta / (enemy.para["move_max_g"] * 9.8) * enemy.para["move_max_speed"]
        return turn_t

    def in_center(self, plane):
        distance_to_center = (plane.X**2 + plane.Y**2 + (plane.Z - 9000)**2)**0.5
        if distance_to_center <= 50000 and plane.Z >= 2000 and plane.Z <= 16000:
            return True
        return False