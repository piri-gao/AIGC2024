import numpy as np
from env.env_cmd import CmdEnv as env_cmd
import random
import math
from utils.utils_math import TSVector3


class Plane(object):
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
        if agent['Heading'] < 0:
            agent['Heading'] += math.pi * 2
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
        if self.Type == 1:
            # 剩余弹药
            self.LeftWeapon = 4
        else:
            self.LeftWeapon = 2
        # 坐标
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        # 被几发导弹瞄准
        self.locked_missile_list = []
        # 已发射导弹
        self.used_missile_list = []
        # 已发射导弹
        self.ready_missile = None
        # 被哪些敌机跟踪
        self.followed_plane = []
        # 比较被敌机跟踪距离
        self.followed_dis = 999999999
        # 敌机最近距离:
        self.close_distance = -1
        # 占据中心时间
        self.center_time = 0
        # 是否丢失
        self.lost_flag = 0
        # 飞机属性参数
        self.para = {}
        # 平台具体属性信息
        if self.Type == 1:
            self.para["move_max_speed"] = 400
            self.para["move_min_speed"] = 150
            self.para["move_max_acc"] = 1
            self.para["move_max_g"] = 6
            self.para["weapon_num"] = 4
            self.para["area_max_alt"] = 15000
            self.para["area_min_alt"] = 2000
            self.para["attack_range"] = 0.8
            self.para['radar_range'] = 80000
            self.para["launch_range"] = 80000 * 0.8
            # self.para["launch_range"] = 60000
            self.para['radar_heading'] = 60
            self.para['radar_pitch'] = 60
        else:
            self.para["move_max_speed"] = 300
            self.para["move_min_speed"] = 100
            self.para["move_max_acc"] = 2
            self.para["move_max_g"] = 12
            self.para["weapon_num"] = 2
            self.para["area_max_alt"] = 10000
            self.para["area_min_alt"] = 2000
            self.para["attack_range"] = 1
            self.para['radar_range'] = 60000
            self.para["launch_range"] = 60000 * 0.8
            # self.para["launch_range"] = 60000
            self.para['radar_heading'] = 30
            self.para['radar_pitch'] = 10

    def __eq__(self, other):
        if other == None:
            return True
        return True if self.ID == other.ID else False

    def update_agent_info(self, agent):
        self.X = agent['X']
        self.Y = agent['Y']
        self.Z = agent['Alt']
        self.Pitch = agent['Pitch']
        if agent['Heading'] < 0:
            agent['Heading'] += math.pi * 2
        self.Heading = agent['Heading']
        self.Roll = agent['Roll']
        self.Speed = agent['Speed']
        self.Availability = agent['Availability']
        self.Type = agent['Type']
        self.lost_flag = 0
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}

    def evade(self, enemy, cmd_list):
        # calclate heading
        if self.Type == 1:
            pitch_ratio = 0.1
            max_pitch = math.pi/6
            heading = self.Heading
        else:
            pitch_ratio = 0.5
            max_pitch = math.pi/5
            heading_diff = enemy.Heading - TSVector3.calheading(TSVector3.normalize(vector))
            if 0 < heading_diff < 2/180*math.pi or (heading_diff < 0 and heading_diff < -2/180*math.pi):
                if self.Heading > math.pi:
                    heading = self.Heading + math.pi/6.5
                else:
                    heading = self.Heading - math.pi/6.5
            else:
                if self.Heading > math.pi:
                    heading = self.Heading - math.pi/6.5
                else:
                    heading = self.Heading + math.pi/6.5
            if heading < 0 :
                heading += math.pi*2
            elif heading > math.pi*2:
                heading -= math.pi*2

        # calculate pitch
        vector = TSVector3.minus(self.pos3d, enemy.pos3d)
        relative_pitch = TSVector3.calpitch(vector)
        if (enemy.Pitch > 0 and enemy.Pitch < relative_pitch * pitch_ratio) or (enemy.Pitch < 0 and enemy.Pitch < relative_pitch * pitch_ratio):
            relative_pitch = max_pitch
        else:
            relative_pitch = -max_pitch
                
        # combine & generate command
        new_dir = TSVector3.calorientation(heading, relative_pitch)
        new_evade_pos = TSVector3.plus(self.pos3d, TSVector3.multscalar(new_dir, 10*self.para["move_max_speed"]))
        if new_evade_pos['Z'] < 3000:
            new_evade_pos['Z'] = self.para["area_max_alt"]-500
        elif new_evade_pos['Z'] > self.para["area_max_alt"]-1000:
            new_evade_pos['Z'] = 2000
        vertical_evade_route_list = [{"X": new_evade_pos['X'], "Y": new_evade_pos['Y'], "Z": new_evade_pos['Z']},]
       
        if abs(new_evade_pos["X"]) > 150000 or abs(new_evade_pos["Y"]) > 150000:
            cmd_list.append(env_cmd.make_areapatrolparam(self.ID, self.X, self.Y, random.random() * 7000 + 2000, 200, 100,
                                                        self.para["move_max_speed"], self.para["move_max_acc"],
                                                        self.para["move_max_g"]))
        else:
            cmd_list.append(
                env_cmd.make_linepatrolparam(self.ID, vertical_evade_route_list, self.para["move_max_speed"],
                                            self.para["move_max_acc"], self.para["move_max_g"]))

    def attackable(self, enemy):
        if self.Availability == 0:
            return False
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        factor = 1.5 if enemy.Type == 1 else 1.375
        return dis*factor < self.para['launch_range']*0.8 and \
            self.in_radar_range(enemy, self.XY2theta(enemy.X - self.X,enemy.Y-self.Y), TSVector3.calpitch(TSVector3.minus(self.pos3d, enemy.pos3d)))

    def visible(self, enemy):
        if self.Availability == 0:
            return False
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        return dis < self.para['radar_range']*0.9 and \
            self.in_radar_range(enemy, self.XY2theta(enemy.X - self.X,enemy.Y-self.Y), TSVector3.calpitch(TSVector3.minus(self.pos3d, enemy.pos3d)))

    def in_radar_range(self, enemy, theta, pitch):
        theta_diff = theta - self.Heading
        while theta_diff > math.pi:
            theta_diff -= 2*math.pi
        while theta_diff <= -math.pi:
            theta_diff += 2*math.pi

        while pitch > math.pi:
            pitch -= 2*math.pi
        while pitch <= -math.pi:
            pitch += 2*math.pi

        return abs(theta_diff)*180/math.pi < self.para['radar_heading'] and \
            abs(pitch)*180/math.pi < self.para['radar_pitch']

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
        # 俯仰角度(浮点型, 单位: 弧度)
        self.Pitch = missile_info['Pitch']
        # 横滚角
        self.Roll = missile_info['Roll']
        # 航向, 即偏航角
        if missile_info['Heading'] < 0:
            missile_info['Heading'] += math.pi * 2
        self.Heading = missile_info['Heading']
        # 来袭导弹发射平台编号
        self.LauncherID = missile_info['LauncherID']
        # 仿真时间
        self.CurTime = missile_info['CurTime']
        # 发射时间
        self.fire_time = missile_info['CurTime']-1
        # 预计到达时间
        self.arrive_time = None
        # 类型
        self.Type = missile_info['Type']
        # 被袭击的平台编号
        self.EngageTargetID = missile_info['EngageTargetID']
        # 军别信息
        self.Identification = missile_info['Identification']
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.lost_flag = 0

    def update_agent_info(self, missile_info):
        self.X = missile_info['X']
        self.Y = missile_info['Y']
        self.Z = missile_info['Alt']
        self.Pitch = missile_info['Pitch']
        if missile_info['Heading'] < 0:
            missile_info['Heading'] += math.pi * 2
        self.Heading = missile_info['Heading']
        self.Roll = missile_info['Roll']
        self.LauncherID = missile_info['LauncherID']
        self.EngageTargetID = missile_info['EngageTargetID']
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.lost_flag = 0
        if self.arrive_time is not None:
            self.arrive_time = self.arrive_time - (self.CurTime - self.fire_time)
 