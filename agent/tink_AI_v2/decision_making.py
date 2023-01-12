import copy
import numpy as np
import math
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3

#定义 一个 TSVector3D
class BaseTSVector3:
    # 初始化
    def __init__(self,x:float,y:float,z:float):
        return {"X": x, "Y":y, "Z": z}

    # 矢量a + 矢量b
    @staticmethod
    def plus(a, b):
        return {"X": a["X"] + b["X"], "Y": a["Y"] + b["Y"], "Z": a["Z"] + b["Z"]}

    # 矢量a - 矢量b
    @staticmethod
    def minus(a, b):
        return {"X": a["X"] - b["X"], "Y": a["Y"] - b["Y"], "Z": a["Z"] - b["Z"]}

    # 矢量a * 标量scal
    @staticmethod
    def multscalar(a, scal):
        return {"X": a["X"] * scal, "Y": a["Y"] * scal, "Z": a["Z"] * scal}

    # 矢量a / 标量scal
    @staticmethod
    def divdbyscalar(a, scal):
        if scal == 0:
            return {"X": 1.633123935319537e+16, "Y": 1.633123935319537e+16, "Z": 1.633123935319537e+16}
        else:
            return {"X": a["X"] / scal, "Y": a["Y"] / scal, "Z": a["Z"] / scal}

    # 矢量a 点乘 矢量b
    @staticmethod
    def dot(a, b):
        return a["X"] * b["X"] + a["Y"] * b["Y"] + a["Z"] * b["Z"]

    # 矢量a 叉乘 矢量b
    @staticmethod
    def cross(a, b):
        val = {"X": a["Y"] * b["Z"] - a["Z"] * b["Y"], \
               "Y": a["Z"] * b["X"] - a["X"] * b["Z"], \
               "Z": a["X"] * b["Y"] - a["Y"] * b["X"]}
        return val

    # 判断矢量a是否为0矢量
    @staticmethod
    def iszero(a):
        if a["X"] == 0 and a["Y"] == 0 and a["Z"] == 0:
            return True
        else:
            return False

    # 矢量a归一化
    @staticmethod
    def normalize(a):
        vallen = math.sqrt(a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"])
        val = {"X": 0, "Y": 0, "Z": 0}
        if vallen > 0:
            val = {"X": a["X"] / vallen, "Y": a["Y"] / vallen, "Z": a["Z"] / vallen}
        return val

    # 计算矢量a的长度
    @staticmethod
    def length(a):
        if a["X"] == 0 and a["Y"] == 0 and a["Z"] == 0:
            return 0
        return math.sqrt(a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"])

    # 计算矢量a的长度平方
    @staticmethod
    def lengthsqr(a):
        return a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"]

class DemoDecision():

    def __init__(self, global_observation):

        self.global_observation = global_observation

        self.init_info()

    def init_info(self):
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
        # 敌方无人机信息列表
        self.enemy_uav_plane = []

        # 我方导弹信息
        self.my_missile = []
        # 敌方导弹信息
        self.enemy_missile = []

        # 被打击的敌机列表
        self.hit_enemy_list = []

        # 识别红军还是蓝军
        self.side = None


        # 有人机第一编队
        self.first_leader_formation = {}
        # 有人机第二编队
        self.sec_leader_formation = {}

        # 无人机第一编队
        self.first_uav_formation = [0, 0]
        # 无人机第二编队
        self.sec_uav_formation = [0, 0]

        # 第一编队
        self.first_formation = {}
        # 第二编队
        self.sec_formation = {}

        # 躲避列表
        self.evade_list = []
        # 干扰列表
        self.need_jam_list = []

    def reset(self):
        """当引擎重置会调用,选手需要重写此方法"""
        self.init_info()

    def update_entity_info(self, sim_time):
        # 己方所有飞机信息
        self.my_plane = self.global_observation.observation.get_all_agent_list()
        # 己方有人机信息
        self.my_leader_plane = self.global_observation.observation.get_agent_info_by_type(1)
        # 己方无人机信息
        self.my_uav_plane = self.global_observation.observation.get_agent_info_by_type(2)
        # 敌方所有飞机信息
        self.enemy_plane = self.global_observation.perception_observation.get_all_agent_list()
        # 敌方有人机信息
        self.enemy_leader_plane = self.global_observation.perception_observation.get_agent_info_by_type(1)
        # 敌方无人机信息
        self.enemy_uav_plane = self.global_observation.perception_observation.get_agent_info_by_type(2)

        # 获取队伍标识
        if self.side is None:
            if self.my_plane[0].Identification == "红方":
                self.side = 1
            else:
                self.side = -1

        # 获取双方导弹信息
        missile = self.global_observation.missile_observation.get_missile_list()
        enemy_missile = []
        my_missile = []
        for rocket in missile:
            # 待测试
            if rocket.Identification == self.my_plane[0].Identification:
                my_missile.append(rocket)
            else:
                enemy_missile.append(rocket)
        self.enemy_missile = enemy_missile
        self.my_missile = my_missile

        # 编队并更新编队信息（待测试）
        # self.formation()

    def formation(self):
        self.first_leader_formation["up_plane"] = self.my_uav_plane[0]
        self.first_leader_formation["down_plane"] = self.my_uav_plane[1]
        self.first_leader_formation["leader"] = self.my_leader_plane[0]
        self.sec_leader_formation["up_plane"] = self.my_uav_plane[2]
        self.sec_leader_formation["down_plane"] = self.my_uav_plane[3]
        self.sec_leader_formation["leader"] = self.my_leader_plane[1]

        self.first_uav_formation[0] = self.my_uav_plane[4]
        self.first_uav_formation[1] = self.my_uav_plane[5]

        self.sec_uav_formation[0] = self.my_uav_plane[6]
        self.sec_uav_formation[1] = self.my_uav_plane[7]

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

    def update_decision(self, sim_time, cmd_list):
        self.update_entity_info(sim_time)
        if sim_time <= 2:
            self.init_pos(sim_time, cmd_list)
        else:
            self.init_move(cmd_list)
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

            for threat_plane in threat_plane_list:
                attack_plane = self.can_attack_plane(threat_plane,evade_plane_id,free_plane)
                if attack_plane is not None:
                    if attack_plane.Type == 1:
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 0.8))
                    else:
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 1))
                    self.hit_enemy_list.append([threat_plane, None])
                    threat_plane.num_locked_missile += 1
                    threat_plane_list.remove(threat_plane)
            
            # 制导
            for enemy_plane in undetected_list:
                dis = 999999
                guide_plane = None
                for my_plane in free_plane:
                    tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                    if tmp_dis < dis and my_plane.can_see(enemy_plane):
                        guide_plane = my_plane
                        dis = tmp_dis

                if guide_plane is not None:
                    free_plane.remove(guide_plane)
                    if 150<enemy_plane.Speed<300:
                        follow_speed = enemy_plane.Speed
                    else:
                        follow_speed = guide_plane.performance()["move_max_speed"]
                        follow_route_list = [enemy_plane.pos3d]
                        cmd_list.append(env_cmd.make_linepatrolparam(guide_plane.ID, follow_route_list, follow_speed,
                                                guide_plane.performance()["move_max_acc"], guide_plane.performance()["move_max_g"]))
                        cmd_list.append(env_cmd.make_followparam(guide_plane.ID, enemy_plane.ID, follow_speed, 1, 6))
            # # 制导
            # if free_plane:
            #     self.find_connect(free_plane,undetected_list)
            #     for my_plane in free_plane:
            #         straight_follow_route_list = self.ans[my_plane.ID]
            #         if straight_follow_route_list != {"X": 0, "Y": 0, "Z": 0}:
            #             distance = 100 * my_plane.performance()["move_max_speed"]
            #             follow_speed = my_plane.performance()["move_max_speed"]
            #             straight_follow_route_list = TSVector3.plus(my_plane.pos3d, straight_follow_route_list)
            #             straight_follow_route_list = [straight_follow_route_list]
            #             cmd_list.append(env_cmd.make_linepatrolparam(my_plane.ID, straight_follow_route_list, follow_speed,
            #                                     my_plane.performance()["move_max_acc"], my_plane.performance()["move_max_g"]))

            # 追击模块
            if len(free_plane)>0 and len(threat_plane_list)>0:
                for my_plane in free_plane:
                    dis = 9999999
                    follow_plane = None
                    if my_plane.LeftWeapon>0 and my_plane.Type!=1:
                        for enemy_plane in threat_plane_list:
                            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                            if tmp_dis < dis:
                                follow_plane = enemy_plane
                                dis = tmp_dis
                    if follow_plane is not None:
                        cmd_list.append(env_cmd.make_followparam(my_plane.ID, follow_plane.ID, my_plane.performance()["move_max_speed"], 1, 6))
                        
            # 躲避模块
            for comba in self.evade_list:
                plane = comba["plane_entity"]
                enemy = comba["missile_entity"]
                plane.evade(enemy, cmd_list)
            # 干扰模块
            if len(need_jam_list):
                self.activate_jam(cmd_list, need_jam_list, sim_time)

    def init_pos(self, sim_time, cmd_list):
        self.formation()
        # 初始化部署
        if sim_time == 2:
            init_direction = 90
            if self.side == -1:
                init_direction = 270
            leader_plane_1 = self.first_leader_formation["leader"]
            leader_plane_2 = self.sec_leader_formation["leader"]
            # 初始化有人机位置
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -145000 * self.side, -75000, 9500, 150, init_direction))
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_2.ID, -145000 * self.side, -75000, 9500, 150, init_direction))

            for key, value in self.first_leader_formation.items():
                if key == "up_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -85000, 9500, 200, init_direction))
                if key == "down_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -65000, 9500, 200, init_direction))

            for key, value in self.sec_leader_formation.items():
                if key == "up_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -65000, 9500, 200, init_direction))
                if key == "down_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -85000, 9500, 200, init_direction))

            for i, plane in enumerate(self.first_uav_formation):
                cmd_list.append(
                    # env_cmd.make_entityinitinfo(plane.ID, -140000 * self.side, 65000 - ((i + 1) * 10000), 9500, 200, init_direction))
                    env_cmd.make_entityinitinfo(plane.ID, -140000 * self.side, -65000, 9500, 200, init_direction))

            for i, plane in enumerate(self.sec_uav_formation):
                cmd_list.append(
                    # env_cmd.make_entityinitinfo(plane.ID, -140000 * self.side, -65000 + ((i + 1) * 10000), 9500, 200, init_direction))
                    env_cmd.make_entityinitinfo(plane.ID, -140000 * self.side, -65000, 9500, 200, init_direction))


    def init_move(self, cmd_list):
        # 无目标移动
        if len(self.enemy_plane) <= 1:
            if len(self.enemy_plane) == 1:
                if self.enemy_plane[0].Y >= 0:
                    for p, plane in self.sec_formation.items():
                        cmd_list.append(env_cmd.make_followparam(plane.ID, self.first_formation[p].ID, 300, 1, 6))
                else:
                    for p, plane in self.first_formation.items():
                        cmd_list.append(env_cmd.make_followparam(plane.ID, self.sec_formation[p].ID, 300, 1, 6))
            else:
                # 没有敌人，朝中心点移动
                first_formation_route = {"up_plane": [{"X": 40000 * self.side, "Y": -8500, "Z": 9500}],
                                         "down_plane": [{"X": 40000 * self.side, "Y": -6500, "Z": 9500}],
                                         "leader_plane": [{"X": 25000 * self.side, "Y": -7500, "Z": 9500}],
                                         "uav_1": [{"X": 30000 * self.side, "Y": -5500, "Z": 9500}],
                                         "uav_2": [{"X": 30000 * self.side, "Y": -4500, "Z": 9500}]
                                         }
                sec_formation_route = {"up_plane": [{"X": 40000 * self.side, "Y": -6500, "Z": 9500}],
                                       "down_plane": [{"X": 40000 * self.side, "Y": -8500, "Z": 9500}],
                                       "leader_plane": [{"X": 25000 * self.side, "Y": -7500, "Z": 9500}],
                                       "uav_1": [{"X": 30000 * self.side, "Y": -5500, "Z": 9500}],
                                       "uav_2": [{"X": 30000 * self.side, "Y": -4500, "Z": 9500}]
                                       }
                for plane, route in first_formation_route.items():
                    cmd_list.append(env_cmd.make_linepatrolparam(self.first_formation[plane].ID, route, 240, 1, 6))
                for plane, route in sec_formation_route.items():
                    cmd_list.append(env_cmd.make_linepatrolparam(self.sec_formation[plane].ID, route, 240, 1, 6))

    def _is_dead(self, plane):
        if plane.Availability == 0:
            return True
        else:
            return False

    def update_evade(self):
        # missile_list = self.global_observation.missile_observation.get_missile_list()
        missile_list = self.enemy_missile
        # missile_id = [missile.ID for missile in missile_list]
        evade_id = [comb["missile_entity"].ID for comb in self.evade_list]
        need_jam_list = []
        # 统计所有被导弹瞄准的飞机
        if len(missile_list) != 0:
            for missile in missile_list:
                attacked_plane = self.global_observation.observation.get_agent_info_by_id(missile.EngageTargetID)
                if attacked_plane is None:
                    continue
                if missile.ID not in evade_id:
                    evade_comb = {}
                    evade_comb["plane_entity"] = attacked_plane
                    evade_comb["missile_entity"] = missile
                    evade_comb["distance"] = TSVector3.distance(missile.pos3d,attacked_plane.pos3d)
                    self.evade_list.append(evade_comb)
            # 给危险程度分类 TODO
        # 将不需要躲避的移除列表
        for attacked_plane in self.evade_list:
            missile = attacked_plane["missile_entity"]
            plane = attacked_plane["plane_entity"]
            distance = attacked_plane["distance"]
            missile_gone, over_target, safe_distance = False, False, False
            # 导弹已爆炸
            if not self.global_observation.missile_observation.get_missile_info_by_rocket_id(missile.ID):
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
            dir = np.math.acos(res) * 180 / np.math.pi
            if abs(dir) > 90:
                over_target = True
            # 飞到安全距离
            if distance > 60000:
                safe_distance = True
            # 需要干扰
            if distance < 60000:
                need_jam_list.append(attacked_plane["plane_entity"])
            if any([missile_gone, over_target, safe_distance]):
                self.evade_list.remove(attacked_plane)
        need_jam_list = sorted(need_jam_list, key=lambda d: d.Type, reverse=True)
        return need_jam_list

    def update_attack(self):
        pass

    def get_threat_target_list(self):
        # 有人机最重要，距离，带弹数量
        threat_dict = {}
        for enemy in self.enemy_plane:
            dis = 99999999
            for my_plane in self.my_leader_plane:
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
        for hit_enemy in self.hit_enemy_list:
            leader_hit = False
            # 敌有人机可以打两发
            if hit_enemy[0].num_locked_missile == 1 and hit_enemy[0].Type == 1:
                leader_hit = True
            for threat_plane in threat_plane_list:
                if hit_enemy[0].ID == threat_plane.ID and not leader_hit:
                    threat_plane_list.remove(threat_plane)
        return threat_plane_list

    def can_attack_plane(self, enemy_plane,evade_plane_id,free_plane):
        
        dis = 9999999
        can_attack_dict = {}
        for my_plane in self.my_plane:
            attack_plane = None
            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
            left_weapon = my_plane.LeftWeapon > 0
            in_range = my_plane.can_attack(tmp_dis)
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
            attack_plane = self.global_observation.observation.get_agent_info_by_id(can_attack_list[0])
        else:
            attack_plane = None      
        return attack_plane

    def update_hit_list(self):
        for comba in self.hit_enemy_list:
            if comba[1] is None:
                for my_missile in self.my_missile:
                    if my_missile.EngageTargetID == comba[0].ID:
                        comba[1] = my_missile
        undetected_list = []
        for comba in self.hit_enemy_list:
            is_dead = True
            if self.global_observation.perception_observation.get_agent_info_by_id(comba[0].ID):
                is_dead = self._is_dead(comba[0])
            missile_gone = True
            if comba[1] is not None:
                if self.global_observation.missile_observation.get_missile_info_by_rocket_id(comba[1].ID):
                    missile_gone = False
            if is_dead or missile_gone:
                self.hit_enemy_list.remove(comba)
            elif not is_dead and not missile_gone:
                if TSVector3.distance(comba[0].pos3d, comba[1].pos3d) > 20000*0.9:
                    undetected_list.append(comba[0])
        undetected_list = sorted(undetected_list, key=lambda d: d.Type, reverse=True)
        return undetected_list

    def activate_jam(self, cmd_list, need_jam_list,sim_time):
        # for comba in self.jam_list:
        #     if sim_time - comba[2]>60 or comba[1]==0 and need_jam_list:
        #         tmp_dir = TSVector3.minus(need_jam_list[0].pos3d,comba[0].pos3d)
        #         total_dir = {"X": 0, "Y": 0, "Z": 0}
        #         for need_jam in need_jam_list:
        #             tmp_dir2 = TSVector3.minus(need_jam.pos3d,comba[0].pos3d)
        #             tmp_dir2 = TSVector3.plus(tmp_dir2,total_dir)
        #             if abs(TSVector3.angle(tmp_dir,tmp_dir2)) < np.math.pi/3:
        #                 total_dir = tmp_dir2.copy()
        #                 need_jam_list.remove(need_jam)
        #         distance = 300 * comba[0].performance()["move_max_speed"]
        #         jam_pos = TSVector3.plus(comba[0].pos3d, total_dir)
        #         straight_jam_route_list = [{"X": jam_pos["X"], "Y": jam_pos["Y"], "Z": jam_pos["Z"]}, ]
        #         comba[1] = 1
        #         comba[2] = sim_time+3
        #         cmd_list.append(
        #             env_cmd.make_linepatrolparam(comba[0].ID, straight_jam_route_list, comba[0].performance()["move_max_speed"], comba[0].performance()["move_max_acc"], comba[0].performance()["move_max_g"]))
        #         cmd_list.append(env_cmd.make_jamparam(comba[0].ID))
        #         break
        #     else:
        #         comba[1] = 0
        can_jam_list = []
        for comba in self.jam_list:
            if sim_time - comba[2]>60 or comba[1]==0:
                can_jam_list.append(comba[0])
            else:
                comba[1] = 0

        if can_jam_list:
            self.find_connect(can_jam_list,need_jam_list)
            for jam in can_jam_list:
                straight_jam_route_list = self.ans[jam.ID]
                if straight_jam_route_list != {"X": 0, "Y": 0, "Z": 0}:
                    for comba in self.jam_list:
                        if comba[0].ID == jam.ID:
                            comba[1] = 1
                            comba[2] = sim_time+3
                            distance = 100 * comba[0].performance()["move_max_speed"]
                            straight_jam_route_list = TSVector3.plus(comba[0].pos3d, straight_jam_route_list)
                            straight_jam_route_list = [straight_jam_route_list]
                            cmd_list.append(
                                env_cmd.make_linepatrolparam(comba[0].ID, straight_jam_route_list, comba[0].performance()["move_max_speed"], comba[0].performance()["move_max_acc"], comba[0].performance()["move_max_g"]))
                            cmd_list.append(env_cmd.make_jamparam(comba[0].ID))

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

    def check_point(self,s,my_plane,e):
        tmp_dir1 = TSVector3.minus(s.pos3d, my_plane.pos3d)
        tmp_dir1 = TSVector3.normalize(tmp_dir1)
        tmp_dir2 = TSVector3.minus(e.pos3d, my_plane.pos3d)
        tmp_dir2 = TSVector3.normalize(tmp_dir2)
        tmp_dir3 = TSVector3.plus(TSVector3.normalize(self.tmp_ans[my_plane.ID]),tmp_dir2)
        if abs(self.angle(tmp_dir1,tmp_dir3)) < np.math.pi/3:
            return TSVector3.normalize(tmp_dir2)
        else:
            return None
    
    def dfs(self,start_enemy_plane,enemy_planes,my_planes,left_num,count):
        if left_num==0:
            len_my_plane_now = 0
            for k,v in self.tmp_ans.items():
                if v != {"X": 0, "Y": 0, "Z": 0}:
                    len_my_plane_now += 1
            if count < self.max_enemy_plane:
                return 
            elif count > self.max_enemy_plane:
                self.max_enemy_plane = count
                self.min_my_plane = len_my_plane_now
                self.ans = self.tmp_ans.copy()
            else:
                if self.min_my_plane<len_my_plane_now:
                    return
                self.min_my_plane = len_my_plane_now
                self.ans = self.tmp_ans.copy()
                return

        for enemy_plane in enemy_planes:
            # if enemy_plane.ID != start_enemy_plane.ID:
            for my_plane in my_planes:
                tmp_dir = self.check_point(start_enemy_plane,my_plane,enemy_plane)
                if self.checked[enemy_plane.ID]==0 and tmp_dir:
                    distance = my_plane.performance()["move_max_speed"] * 100
                    self.tmp_ans[my_plane.ID] = TSVector3.plus(self.tmp_ans[my_plane.ID], TSVector3.multscalar(tmp_dir, distance))
                    self.checked[enemy_plane.ID] = 1
                    self.dfs(start_enemy_plane,enemy_planes,my_planes,left_num-1,count+1)
                    self.checked[enemy_plane.ID] = 0
                    self.tmp_ans[my_plane.ID] = TSVector3.minus(self.tmp_ans[my_plane.ID], TSVector3.multscalar(tmp_dir, distance))

    def find_connect(self, my_planes, enemy_planes):
        self.ans = {}
        for my_plane in my_planes:
            self.ans[my_plane.ID] = {"X": 0, "Y": 0, "Z": 0}
        self.min_my_plane = len(my_planes)
        self.max_enemy_plane = 0
        self.tmp_ans = {}
        self.checked = {}
        for enemy_plane in enemy_planes:
            for enemy_plane in enemy_planes:
                self.checked[enemy_plane.ID] = 0 
            for my_plane in my_planes:
                self.tmp_ans[my_plane.ID] = {"X": 0, "Y": 0, "Z": 0}
            self.dfs(enemy_plane,enemy_planes,my_planes,len(enemy_planes),1)