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
            self.last_jam = 0
            self.in_jam = False
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

        self.have_move_cmd = False

        self.jammed = 0
        self.closest_missile = None
        self.jam_plane = None

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
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        vector = TSVector3.minus(self.pos3d, enemy.pos3d)
        relative_pitch = TSVector3.calpitch(vector)
        dir = TSVector3.normalize(vector)
        enemy_pitch = enemy.Pitch

        if self.Type == 1:
            if enemy == None:
                return
            relative_pitch = math.pi / 5.5 if self.CurTime % 2 else -math.pi / 5.5
            heading = TSVector3.calheading(dir)
            if abs(self.bound_in_pi(self.Heading- heading - math.pi / 2)) > math.pi / 2:
                heading = heading - math.pi / 2
            else:
                heading = heading + math.pi / 2

            if abs(self.bound_in_pi(self.Heading - heading)) < math.pi / 5 or dis < 5000:
                turn_time = 0
            else:
                turn_time = abs(self.bound_in_pi(self.Heading - heading)) / (self.para['move_max_g'] * 9.8) * self.Speed
            if dis / 1000 > (self.para["move_max_speed"] - self.Speed) / 9.8 + turn_time + 10:
                move_speed = self.Speed
            else:
                move_speed = self.para["move_max_speed"]
            dis_factor = 20
        else:
            if (enemy_pitch > 0 and enemy_pitch < relative_pitch * 0.5) or (enemy_pitch < 0 and enemy_pitch < relative_pitch * 0.5):
                relative_pitch = math.pi / 5
            else:
                relative_pitch = - math.pi / 5 
            alt = random.random() * 7000 + 2000
            heading = self.Heading
            pi_ratio = 1/2
            if 3000 < dis < 10000:
                if abs(self.bound_in_pi(self.Heading - enemy.Heading - math.pi / 2)) > math.pi / 2:
                    heading = enemy.Heading - math.pi / 2
                else:
                    heading = enemy.Heading + math.pi / 2
            dis_factor = 10
                
        distance = dis_factor * self.para["move_max_speed"]
        new_dir = TSVector3.calorientation(heading, relative_pitch)   
        new_evade_pos = TSVector3.plus(self.pos3d, TSVector3.multscalar(new_dir, distance))
        new_plane_pos = TSVector3.plus(self.pos3d, TSVector3.multscalar(new_dir, dis / enemy.Speed * self.Speed))
        if new_evade_pos['Z'] < 2000 or (self.Type == 1 and self.Z < 2800):
            new_evade_pos['Z'] = self.Z + math.sin(abs(relative_pitch)) * distance
        elif new_evade_pos['Z'] > self.para["area_max_alt"] or (self.Type == 1 and self.Z > self.para["area_max_alt"] - 800):
            new_evade_pos['Z'] = self.Z - math.sin(abs(relative_pitch)) * distance
        vertical_evade_route_list = [{"X": new_evade_pos['X'], "Y": new_evade_pos['Y'], "Z": new_evade_pos['Z']}, ]

        if self.Type == 1:
            if (abs(new_evade_pos["X"]) >= 142000 or abs(new_evade_pos["Y"]) >= 142000):
                x = self.X
                y = self.Y
                if abs(self.X)+1000>=150000 or abs(self.Y)+1000>=150000:
                    x = -1*self.X/abs(self.X)*200+self.X
                    y = -1*self.Y/abs(self.Y)*200+self.Y
                cmd_list.append(env_cmd.make_areapatrolparam(self.ID, x, y, vertical_evade_route_list[0]["Z"], move_speed, 100, self.para["move_max_speed"],
                                            self.para["move_max_acc"], self.para["move_max_g"]))
                self.have_move_cmd = True
                # cmd_list.append(env_cmd.make_areapatrolparam(self.ID, self.X, self.Y, vertical_evade_route_list[0]["Z"], move_speed, 100, self.para["move_max_speed"],
                #                             self.para["move_max_acc"], self.para["move_max_g"]))
                
            if (abs(new_evade_pos["X"]) < 142000 and abs(new_evade_pos["Y"]) < 142000):
                self.have_move_cmd = True
                cmd_list.append(env_cmd.make_linepatrolparam(self.ID, vertical_evade_route_list,
                                            move_speed,
                                            self.para["move_max_acc"], self.para["move_max_g"]))
        else:
            if abs(new_evade_pos["X"]) > 142000 or abs(new_evade_pos["Y"]) > 142000:
                x = self.X
                y = self.Y
                if abs(self.X)+1000>=150000 or abs(self.Y)+1000>=150000:
                    x = -1*self.X/abs(self.X)*200+self.X
                    y = -1*self.Y/abs(self.Y)*200+self.Y
                cmd_list.append(env_cmd.make_areapatrolparam(self.ID, x, y, alt, 200, 100,
                                                            self.para["move_max_speed"], self.para["move_max_acc"],
                                                            self.para["move_max_g"]))
                self.have_move_cmd = True
                # cmd_list.append(env_cmd.make_areapatrolparam(self.ID, self.X, self.Y, alt, 200, 100,
                #                                             self.para["move_max_speed"], self.para["move_max_acc"],
                #                                             self.para["move_max_g"]))
            else:
                self.have_move_cmd = True
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
        theta_diff = self.bound_in_pi(theta - self.Heading)
        pitch = self.bound_in_pi(pitch)
        return abs(theta_diff)*180/math.pi < self.para['radar_heading'] and \
            abs(pitch)*180/math.pi < self.para['radar_pitch']

    def bound_in_pi(self, theta):
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta <= - math.pi:
            theta += 2 * math.pi
        return theta

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
        # 航速
        self.Speed = missile_info['Speed']
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
        self.Speed = missile_info['Speed']
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
 