import numpy as np
from env.env_cmd import CmdEnv as env_cmd
import random
import math
from utils.utils_math import TSVector3


class MyPlane(object):
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

        # 坐标
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        # 被几发导弹瞄准
        self.num_locked_missile = 0

    def __eq__(self, other):
        if other == None:
            return True
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
        self.IsLocked = agent['IsLocked']
        self.LeftWeapon = agent['LeftWeapon']
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        
    def performance(self):
        para = {}
        if self.Type == 1:
            para["move_max_speed"] = 400
            para["move_min_speed"] = 150
            para["move_max_acc"] = 1
            para["move_max_g"] = 6
            para["weapon_num"] = 4
            para["area_max_alt"] = 15000
            para["area_min_alt"] = 2000
            para["attack_range"] = 0.8
            para['radar_range'] = 80000
            para["launch_range"] = 60000 * 0.8
            para['radar_heading'] = 60
            para['radar_pitch'] = 60
        else:
            para["move_max_speed"] = 300
            para["move_min_speed"] = 100
            para["move_max_acc"] = 2
            para["move_max_g"] = 12
            para["weapon_num"] = 2
            para["area_max_alt"] = 10000
            para["area_min_alt"] = 2000
            para["attack_range"] = 1
            para['radar_range'] = 60000
            para["launch_range"] = 60000 * 0.8
            para['radar_heading'] = 30
            para['radar_pitch'] = 10
        return para

    def evade(self, enemy, cmd_list):
        EntityPos = {"X": self.X, "Y": self.Y, "Z": self.Z}
        EnemyPos = {"X": enemy.X, "Y": enemy.Y, "Z": enemy.Z}
        performance = self.performance()
        dis = TSVector3.distance(EntityPos, EnemyPos)
        vector = TSVector3.minus(EntityPos, EnemyPos)
        
        relative_pitch = TSVector3.calpitch(vector)
        enemy_pitch = enemy.Pitch/180*np.math.pi
        if abs(enemy_pitch - relative_pitch)<np.math.pi/18:
            relative_pitch = enemy_pitch
        else:
            relative_pitch = 0
        # 选择与敌方导弹相反为仰角
        # dir = TSVector3.calorientation(enemy.Heading, relative_pitch)
        dir = TSVector3.normalize(vector)
        alt = random.random()*7000 + 2000
        distance = 300 * performance["move_max_speed"]
        evade_pos = TSVector3.plus(EntityPos, TSVector3.multscalar(dir, distance))
        if evade_pos['Z']<2000 or evade_pos['Z'] > performance["area_max_alt"]:
            evade_pos['Z'] = alt
        straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": evade_pos["Z"]}, ]
        # 选择90度后的单位
        dir = TSVector3.normalize(vector)
        heading = TSVector3.calheading(dir)
        new_heading = heading + np.math.pi / 2
        # if self.Heading < enemy.Heading:
        #     new_heading = enemy.Heading - np.math.pi/2
        # else:
        #     new_heading = enemy.Heading + np.math.pi/2
        # 选择与敌方导弹相反为仰角
        new_dir = TSVector3.calorientation(new_heading, relative_pitch)
        new_evade_pos = TSVector3.plus(EntityPos, TSVector3.multscalar(new_dir, distance))
        if new_evade_pos['Z']<2000 or new_evade_pos['Z'] > performance["area_max_alt"]:
            new_evade_pos['Z'] = alt
        vertical_evade_route_list = [{"X": new_evade_pos['X'], "Y": new_evade_pos['Y'], "Z": new_evade_pos['Z']}, ]

        if dis > 25000:
            # 盘旋或反方向远离敌方位置
            if abs(evade_pos["X"]) > 150000 or abs(evade_pos["Y"]) > 150000:
                cmd_list.append(
                    env_cmd.make_areapatrolparam(self.ID, self.X, self.Y, evade_pos['Z'], 200, 100, performance["move_max_speed"],
                                                performance["move_max_acc"], performance["move_max_g"]))
            else:
                cmd_list.append(
                    env_cmd.make_linepatrolparam(self.ID, straight_evade_route_list, performance["move_max_speed"],
                                                performance["move_max_acc"], performance["move_max_g"]))
        else:
            # 盘旋或垂直飞行使敌方导弹脱靶
            if abs(new_evade_pos["X"]) > 150000 or abs(new_evade_pos["Y"]) > 150000:
                cmd_list.append(env_cmd.make_areapatrolparam(self.ID, self.X, self.Y, alt, 200, 100,
                                                            performance["move_max_speed"], performance["move_max_acc"],
                                                            performance["move_max_g"]))
            else:
                cmd_list.append(
                    env_cmd.make_linepatrolparam(self.ID, vertical_evade_route_list, performance["move_max_speed"],
                                                performance["move_max_acc"], performance["move_max_g"]))

    def can_attack(self, enemy):
        # performance = self.performance()
        # if performance["launch_range"] > dis:
        #     return True
        # return False
        performance = self.performance()
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        enemy_theta = self.XY2theta(enemy.X - self.X,enemy.Y-self.Y)
        vector = TSVector3.minus(self.pos3d, enemy.pos3d)
        relative_pitch = TSVector3.calpitch(vector)*180/math.pi
        if dis < performance['launch_range'] and \
            abs(self.pi_bound(enemy_theta - self.Heading))<performance['radar_heading'] and \
            abs(self.pi_bound(relative_pitch))<performance['radar_pitch']:
            return True
        return False

    def can_see(self, enemy):
        performance = self.performance()
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        enemy_theta = self.XY2theta(enemy.X - self.X,enemy.Y-self.Y)
        vector = TSVector3.minus(self.pos3d, enemy.pos3d)
        relative_pitch = TSVector3.calpitch(vector)*180/math.pi
        if dis < performance['radar_range']*0.8 and \
            abs(self.pi_bound(enemy_theta - self.Heading))<performance['radar_heading'] and \
            abs(self.pi_bound(relative_pitch))<performance['radar_pitch']:
            return True
        return False

    def XY2theta(self, X, Y):
        theta = 0
        if Y == 0:
            if X >= 0:
                theta = math.pi / 2
            else:
                theta = -math.pi / 2
        else:
            theta = math.atan(X / Y)
            if Y < 0:
                if X >= 0:
                    theta += math.pi
                else:
                    theta -= math.pi
        return theta

    def pi_bound(self, theta):
        while theta > math.pi:
            theta -= 2*math.pi
        while theta <= -math.pi:
            theta += 2*math.pi
        return theta

class EnemyPlane(object):
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
        # 坐标
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        # 被几发导弹瞄准
        self.num_locked_missile = 0

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
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}


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
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}

    def update_missile_info(self, missile_info):
        self.X = missile_info['X']
        self.Y = missile_info['Y']
        self.Z = missile_info['Alt']
        self.Pitch = missile_info['Pitch']
        self.Heading = missile_info['Heading']
        self.Roll = missile_info['Roll']
        self.LauncherID = missile_info['LauncherID']
        self.EngageTargetID = missile_info['EngageTargetID']
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}