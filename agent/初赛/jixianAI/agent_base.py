""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/8/9 15:14
"""
from env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
from agent.jixianAI.utils import reg_rad, dir2rad, adjust_angle_to_target, check_dis
import numpy as np
import copy

class MyJet(object):
    def __init__(self, agent):
        # 平台编号
        self.ID = agent['ID']
        # x轴坐标(浮点型, 单位: 米, 下同)
        self.X = agent['X']
        # y轴坐标
        self.Y = agent['Y']
        # z轴坐标
        self.Z = agent['Alt']
        # 航向(浮点型, 单位: 度, [0-360])
        self.Pitch = agent['Pitch']
        # 横滚角
        self.Roll = agent['Roll']
        # 航向, 即偏航角
        self.Heading = agent['Heading']
        # 速度
        self.Speed = agent['Speed']
        # 当前可用性
        self.Availability = agent['Availability']
        # 类型
        self.Type = agent['Type']
        # 仿真时间
        self.CurTime = agent['CurTime']
        # 军别信息
        self.Identification = agent['Identification']
        # 是否被锁定
        self.IsLocked = agent['IsLocked']
        # 剩余弹药
        self.LeftWeapon = agent['LeftWeapon']
        # 名字
        self.Name = agent["Name"]

        # 坐标合集
        self.pos2d_dic = {"X": self.X, "Y": self.Y}
        self.pos3d_dic = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.pos2d = np.array([self.X, self.Y], dtype=float)
        self.pos3d = np.array([self.X, self.Y, self.Z], dtype=float)

        # 被几发导弹瞄准
        self.num_locked_missile = 0
        # 被瞄准的导弹列表
        self.attacked_missile_list = []
        # 上一个步长瞄准我的导弹数
        self.previous_attacked_missile_list = []
        # 发射的导弹列表
        self.my_launch_missile = []
        # 在我雷达探测范围内，且满足开火条件的敌机
        self.in_attack_rader = []
        # 是否为持续执行状态,若是的话需要执行完当前任务才可以执行新的动作[任务，对应的目标]
        self.persistent_status = None
        # 飞机当前执行的动作
        self.current_status = None

        self.alive = False



        self.init_battle_info()
        self.static_para()

    def __eq__(self, other):
        return True if self.ID == other.ID else False

    def init_battle_info(self):
        # 初始化jet所属的编队，待测试
        formation_1st = ["红有人机1", "红无人机1", "红无人机2", "红无人机3", "红无人机4",
                         "蓝有人机1", "蓝无人机1", "蓝无人机2", "蓝无人机3", "蓝无人机4"]
        formation_2nd = ["红有人机2", "红无人机5", "红无人机6", "红无人机7", "红无人机8",
                         "蓝有人机2", "蓝无人机5", "蓝无人机6", "蓝无人机7", "蓝无人机8"]
        if self.Name in formation_1st:
            self.formation = "first"
        else:
            self.formation = "second"


    def update_agent_info(self, agent):
        self.X = agent['X']
        self.Y = agent['Y']
        self.Z = agent['Alt']
        self.Pitch = agent['Pitch']
        self.Heading = agent['Heading']
        self.Roll = agent['Roll']
        self.Speed = agent['Speed']
        self.Availability = agent['Availability']
        self.Type = agent['Type']
        self.IsLocked = agent['IsLocked']
        self.LeftWeapon = agent['LeftWeapon']
        self.pos2d_dic = {"X": self.X, "Y": self.Y}
        self.pos3d_dic = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.pos2d = np.array([self.X, self.Y])
        self.pos3d = np.array([self.X, self.Y, self.Z])


        # 更新探测敌机信息情况
        # 雷达探测分为两种距离，有人机举例，雷达探测到可攻击是60e3，较好的开火距离是58e3
        # 更新可攻击距离雷达探测情况
        #self.in_attack_rader = self.get_in_attack_rader_enemy()
        # 更新较好开火距离内雷达探测情况
        #self.in_rader = self.get_in_rader_enemy()

        self.previous_attacked_missile_list = copy.copy(self.attacked_missile_list)
        # 清零，然后在decision——making里面更新
        self.attacked_missile_list = []

        self.alive = False


    def performance(self):
        para = {}
        if self.Type == 1:
            para["move_max_speed"] = 400
            para["move_max_acc"] = 1
            para["move_max_g"] = 6
            para["area_max_alt"] = 14000
            para["attack_range"] = 1
            para["launch_range"] = 80000
        else:
            para["move_max_speed"] = 300
            para["move_max_acc"] = 2
            para["move_max_g"] = 12
            para["area_max_alt"] = 10000
            para["attack_range"] = 1
            para["launch_range"] = 60000

        # 有人机和无人机，开火距离是多少，是不是有目标就可以发出去，指令下单了距离不够会立即发弹吗
        # 开火角度是180，
        return para

    def static_para(self):
        if self.Type == 1:
            self.MIN_X = -145e3
            self.MAX_X = 145e3
            self.MIN_Y = -145e3
            self.MAX_Y = 145e3

            self.max_speed = 400
            self.min_speed = 150
            self.max_acc = 1
            self.max_g = 6
            self.max_alt = 14000
            self.min_alt = 2000
            self.attack_range = 1
            self.launch_range = 80e3

            self.Raderdis = 58e3
            self.RadarHorizon = 50
            self.RadarVertical = 50

            self.AttackRadarDis = 60e3
            self.AttackRadarHorizon = 50
            self.AttackRadarVertical = 50
        else:
            self.MIN_X = -145e3
            self.MAX_X = 145e3
            self.MIN_Y = -145e3
            self.MAX_Y = 145e3

            self.max_speed = 300
            self.max_acc = 2
            self.max_g = 12
            self.max_alt = 10000
            self.min_alt = 2000
            self.attack_range = 1
            self.launch_range = 60e3

            self.Raderdis = 39e3
            self.RadarHorizon = 30
            self.RadarVertical = 10

            self.AttackRadarDis = 45e3
            self.AttackRadarHorizon = 50
            self.AttackRadarVertical = 50


    def evade(self, enemy, cmd_list):
        EntityPos = {"X": self.X, "Y": self.Y, "Z": self.Z}
        EnemyPos = {"X": enemy.X, "Y": enemy.Y, "Z": enemy.Z}

        performance = self.performance()

        dis = TSVector3.distance(EntityPos, EnemyPos)

        if EntityPos["Z"] >= 2000 and EntityPos["Z"] <= (performance["area_max_alt"] + 2000) / 2.0:
            alt = 2000
        else:
            alt = performance["area_max_alt"]

        EnemyPos["Z"] = EntityPos["Z"]
        vector = TSVector3.minus(EntityPos, EnemyPos)
        dir = TSVector3.normalize(vector)
        distance = 300 * performance["move_max_speed"]
        evade_pos = TSVector3.plus(EntityPos, TSVector3.multscalar(dir, distance))
        straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": EnemyPos["Z"]}, ]

        # 选择90度后的单位
        heading = TSVector3.calheading(dir)
        new_heading = heading + np.math.pi / 2
        new_dir = TSVector3.calorientation(new_heading, 0)
        new_evade_pos = TSVector3.plus(EntityPos, TSVector3.multscalar(new_dir, distance))
        vertical_evade_route_list = [{"X": new_evade_pos['X'], "Y": new_evade_pos['Y'], "Z": alt}, ]

        if dis > 20000:
            # 计算规避位置
            if abs(evade_pos["X"]) > 150000 or abs(evade_pos["Y"]) > 150000 or evade_pos["Z"] > performance["area_max_alt"] or \
                    evade_pos["Z"] < 2000:
                cmd_list.append(
                    CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, alt, 200, 100, performance["move_max_speed"],
                                                performance["move_max_acc"], performance["move_max_g"]))
            else:
                cmd_list.append(
                    CmdEnv.make_linepatrolparam(self.ID, straight_evade_route_list, performance["move_max_speed"],
                                                performance["move_max_acc"], performance["move_max_g"]));
        else:
            # 盘旋飞行使敌方导弹脱靶
            if dis >= 58000:
                if abs(new_evade_pos["X"]) > 150000 or abs(new_evade_pos["Y"]) > 150000:
                    cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, alt, 200, 100,
                                                                performance["move_max_speed"], performance["move_max_acc"],
                                                                performance["move_max_g"]))
                else:
                    cmd_list.append(
                        CmdEnv.make_linepatrolparam(self.ID, vertical_evade_route_list, performance["move_max_speed"],
                                                    performance["move_max_acc"], performance["move_max_g"]))
            else:
                cmd_list.append(
                    CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, alt, 200, 100, performance["move_max_speed"],
                                                performance["move_max_acc"], performance["move_max_g"]))

    def close_target_tracking(self, target):
        dest_location = adjust_angle_to_target(my_jet=self, vip=target, angle=110)
        return dest_location
    '''
    def abort_attack(self, force=False):
        """
        放弃攻击的条件：
            1. 这条指令是攻击指令
            2. 这条指令是发送给当前飞机的
            3.

        """
        def confirm_abort(target_id):
            if self.Type == 1:
                _, threat_distance = self.get_nearest_threat(self.enemy_jets)
                if threat_distance > 30e3:
                    return False
                else:
                    return True
            if force:
                return True

            enemy_jet = self.find_jet_by_given(target_id, True)
            ms = self.get_nearest_ms()
            if ms is None:
                return False

            distance_to_ms = self.get_target_dis(ms)
            # 如果要打击的敌人在我的雷达探测范围内（开火角度内）并且攻击我的导弹距离大于6公里，那么不放弃攻击
            if (enemy_jet in self.in_attack_rader) and (distance_to_ms>6e3):
                return False
            else:
                return True

        cmd_list_new = []
        for cmd in self.cmd_list:
            abort_condition = ('CmdAttackControl' in cmd) and (
                        cmd['CmdAttackControl']['HandleID'] == self.ID) and confirm_abort(
                cmd['CmdAttackControl']['TgtID'])

            if abort_condition:
                pass
            else:
                cmd_list_new.append(cmd)
        return cmd_list_new
    '''

    def get_target_dis(self, target):
        pos1 = self.pos3d_dic
        pos2 = target.pos3d_dic
        return TSVector3.distance(pos1, pos2)

    def get_nearest_threat(self, enemy_jets):
        # 获取最近的威胁目标，可能是导弹或敌机
        threat_target = None
        threat_dis = 9999999
        nearest_ms = self.get_nearest_ms()
        nearest_jet = self.get_nearest_enemy_jet(enemy_jets)

        if nearest_ms is not None and threat_dis > self.get_target_dis(nearest_ms):
            threat_target = nearest_ms
            threat_dis = self.get_target_dis(nearest_ms)
        if nearest_jet is not None and threat_dis > self.get_target_dis(nearest_jet):
            threat_target = nearest_jet
            threat_dis = self.get_target_dis(nearest_jet)
        return threat_target, threat_dis

    def get_nearest_ms(self):
        if len(self.attacked_missile_list) == 0:
            return None
        index = np.argmin(np.array([self.get_target_dis(ms)]) for ms in self.attacked_missile_list)
        return self.attacked_missile_list[index]

    def get_nearest_enemy_jet(self, enemy_jets):
        dis = np.array([self.get_target_dis(enemy_jet) for enemy_jet in enemy_jets if enemy_jet.EnemyLeftWeapon > 0])
        if len(dis) > 0:
            return enemy_jets[np.argmin(dis)]
        else:
            return None

    def get_in_attack_rader_enemy(self):
        in_rader = []
        for enemy_jet in self.enemy_jets:
            dis = self.get_target_dis(enemy_jet)
            if dis > self.AttackRadarDis:
                continue

            delta = enemy_jet.pos2d - self.pos2d
            theta = 90 - self.Heading * 180 / np.pi
            theta = reg_rad(theta * np.pi / 180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -self.AttackRadarHorizon and deg <= self.AttackRadarHorizon:
                in_rader.append(enemy_jet)

        return in_rader

    def get_in_rader_enemy(self):
        in_rader = []
        for enemy_jet in self.enemy_jets:
            dis = self.get_target_dis(enemy_jet)
            if dis > self.attack_dis:
                continue

            delta = enemy_jet.pos2d - self.pos2d
            theta = 90 - self.Heading * 180 / np.pi
            theta = reg_rad(theta * np.pi / 180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -self.AttackRadarHorizon and deg <= self.AttackRadarHorizon:
                in_rader.append(enemy_jet)

        return in_rader

    def check_ms_number(self):
        return len(self.previous_attacked_missile_list) != len(self.attacked_missile_list)


class EnemyJet(object):
    def __init__(self, agent):
        # 平台编号
        self.ID = agent['ID']
        # x轴坐标(浮点型, 单位: 米, 下同)
        self.X = agent['X']
        # y轴坐标
        self.Y = agent['Y']
        # z轴坐标
        self.Z = agent['Alt']
        # 航向(浮点型, 单位: 度, [0-360])
        self.Pitch = agent['Pitch']
        # 横滚角
        self.Roll = agent['Roll']
        # 航向, 即偏航角
        self.Heading = agent['Heading']
        # 速度
        self.Speed = agent['Speed']
        # 当前可用性
        self.Availability = agent['Availability']
        # 类型
        self.Type = agent['Type']
        # 仿真时间
        self.CurTime = agent['CurTime']
        # 军别信息
        self.Identification = agent['Identification']
        # 名字
        self.Name = agent["Name"]

        # 坐标
        self.pos2d_dic = {"X": self.X, "Y": self.Y}
        self.pos3d_dic = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.pos2d = np.array([self.X, self.Y])
        self.pos3d = np.array([self.X, self.Y, self.Z])
        # 被几发导弹瞄准
        self.num_locked_missile = 0
        # 发射的导弹列表
        self.my_launch_missile = []
        self.attacked_missile_list = []
        self.previous_attacked_missile_list = []

        self.alive = False

        if self.Type == 1:
            self.EnemyLeftWeapon = 4
        else:
            self.EnemyLeftWeapon = 2

    def __eq__(self, other):
        return True if self.ID == other.ID else False

    def update_agent_info(self, agent):
        self.X = agent['X']
        self.Y = agent['Y']
        self.Z = agent['Alt']
        self.Pitch = agent['Pitch']
        self.Heading = agent['Heading']
        self.Roll = agent['Roll']
        self.Speed = agent['Speed']
        self.Availability = agent['Availability']
        self.Type = agent['Type']
        self.pos2d_dic = {"X": self.X, "Y": self.Y}
        self.pos3d_dic = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.pos2d = np.array([self.X, self.Y])
        self.pos3d = np.array([self.X, self.Y, self.Z])

        self.previous_attacked_missile_list = copy.copy(self.attacked_missile_list)
        # 清零，然后在decision——making里面更新
        self.attacked_missile_list = []

        self.alive = False



class Missile(object):
    def __init__(self, missile_info):
        # 导弹编号
        self.ID = missile_info['ID']
        # 导弹位置x轴坐标(浮点型, 单位: 米)
        self.X = missile_info['X']
        # 导弹位置y轴坐标
        self.Y = missile_info['Y']
        # 导弹位置z轴坐标
        self.Z = missile_info['Alt']
        # 俯仰角度(浮点型, 单位: 度)
        self.Pitch = missile_info['Pitch']
        # 横滚角
        self.Roll = missile_info['Roll']
        # 航向, 即偏航角
        self.Heading = missile_info['Heading']
        # 来袭导弹发射平台编号
        self.LauncherID = missile_info['LauncherID']
        # 仿真时间
        self.CurTime = missile_info['CurTime']
        # 类型
        self.Type = missile_info['Type']
        # 被袭击的平台编号
        self.EngageTargetID = missile_info['EngageTargetID']
        # 军别信息
        self.Identification = missile_info['Identification']
        ## 新增导弹的可用性
        self.Availability = missile_info['Availability']

        self.pos2d_dic = {"X": self.X, "Y": self.Y}
        self.pos3d_dic = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.pos2d = np.array([self.X, self.Y])
        self.pos3d = np.array([self.X, self.Y, self.Z])

        self.flying_time = 0

    def update_missile_info(self, missile_info):
        self.X = missile_info['X']
        self.Y = missile_info['Y']
        self.Z = missile_info['Alt']
        self.Pitch = missile_info['Pitch']
        self.Heading = missile_info['Heading']
        self.Roll = missile_info['Roll']
        self.LauncherID = missile_info['LauncherID']
        self.EngageTargetID = missile_info['EngageTargetID']
        self.pos2d_dic = {"X": self.X, "Y": self.Y}
        self.pos3d_dic = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.pos2d = np.array([self.X, self.Y])
        self.pos3d = np.array([self.X, self.Y, self.Z])
        self.flying_time += 1
        # 如果没有被重置为True，那就是爆炸了
        self.alive = False