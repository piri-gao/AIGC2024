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
        # 航向(浮点型)
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
        # 僚机
        self.wing_plane = None
        # 仿真时间
        self.CurTime = agent['CurTime']
        # 军别信息
        self.Identification = agent['Identification']
        if self.Type == 1:
            # 剩余弹药
            self.LeftWeapon = 4
            # 上一次执行干扰时间
            self.last_jam = 0
            # 是否在执行干扰任务
            self.do_jam = False
            # 已经被提供了干扰
            self.jammed = 0
            # 最近的一枚导弹
            self.close_missile = None
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
        # 被几枚导弹攻击过了
        self.attacked_missile_num = 0
        # 被哪些敌机跟踪
        self.followed_plane = []
        # 跟踪哪个敌机
        self.follow_plane = None
        # 干扰中间的敌机
        self.middle_enemy_plane = None
        # 是否有无导弹飞机跟踪
        self.have_no_missile_plane = None
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
        # 上一次躲避导弹方向
        self.turn_flag = 1
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
            self.para["launch_range"] = 15000
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
            self.para["launch_range"] = 15000
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
        self.CurTime = agent['CurTime']
        self.Availability = agent['Availability']
        self.Type = agent['Type']
        self.lost_flag = 0
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}

    def evade(self, enemy, cmd_list):# 无人机躲避导弹策略
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        vector = TSVector3.minus(self.pos3d, enemy.pos3d)
        relative_pitch = TSVector3.calpitch(vector)
        enemy_pitch = enemy.Pitch
        if (enemy_pitch > 0 and enemy_pitch < relative_pitch*0.5) or (enemy_pitch < 0 and enemy_pitch < relative_pitch*0.5):
            relative_pitch = math.pi/5
        else:
            relative_pitch = -math.pi/5 
        alt = random.random()*7000 + 2000
        distance = 10*self.para["move_max_speed"]
        heading = self.Heading
        if dis<10000:
            if abs(self.pi_bound(self.Heading - enemy.Heading - math.pi/2))>math.pi/2:
                heading = enemy.Heading - math.pi/2
            else:
                heading = enemy.Heading + math.pi/2
        new_dir = TSVector3.calorientation(heading, relative_pitch)
        new_evade_pos = TSVector3.plus(self.pos3d, TSVector3.multscalar(new_dir, distance))
        if new_evade_pos['Z'] < 3000:
            new_evade_pos['Z'] = self.para["area_max_alt"]-500
        elif new_evade_pos['Z'] > self.para["area_max_alt"]-1000:
            new_evade_pos['Z'] = 2000
        vertical_evade_route_list = [{"X": new_evade_pos['X'], "Y": new_evade_pos['Y'], "Z": new_evade_pos['Z']},]
        # 盘旋或垂直飞行使敌方导弹脱靶
        if abs(new_evade_pos["X"]) > 150000 or abs(new_evade_pos["Y"]) > 150000:
            cmd_list.append(env_cmd.make_areapatrolparam(self.ID, self.X, self.Y, alt, 200, 100,
                                                        self.para["move_max_speed"], self.para["move_max_acc"],
                                                        self.para["move_max_g"]))
        else:
            cmd_list.append(
                env_cmd.make_linepatrolparam(self.ID, vertical_evade_route_list, self.para["move_max_speed"],
                                            self.para["move_max_acc"], self.para["move_max_g"]))

    def evade_leader(self, enemy, cmd_list):# 有人机躲避导弹策略
        if enemy==None:
            return
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        vector = TSVector3.minus(self.pos3d, enemy.pos3d)
        relative_pitch = TSVector3.calpitch(vector)
        enemy_pitch = enemy.Pitch
        dir = TSVector3.normalize(vector)
        if self.CurTime%2:
            relative_pitch = math.pi/5.5
        else:
            relative_pitch = -math.pi/5.5
        # 选择90度后的单位
        heading = TSVector3.calheading(dir)
        if abs(self.pi_bound(self.Heading- heading - math.pi/2))>math.pi/2:
            new_heading = heading - math.pi / 2
        else:
            new_heading = heading + math.pi / 2
        if abs(self.pi_bound(self.Heading - new_heading))<math.pi/5 or dis<5000:
            new_heading = self.Heading
            turn_time = 0
        else:
            turn_time = abs(self.pi_bound(self.Heading - new_heading)) / (self.para['move_max_g'] * 9.8) * self.Speed
        if relative_pitch>0:
            distance, relative_pitch = self.limit_height(relative_pitch, self.para["area_max_alt"])
        else:
            distance, relative_pitch = self.limit_height(relative_pitch, self.Z-2000)
        new_dir = TSVector3.calorientation(new_heading, relative_pitch)
        if dis/1000>(self.para["move_max_speed"]-self.Speed)/9.8 + 10 + turn_time:
            if self.Speed>self.para["move_max_speed"]:
                move_speed = self.para["move_max_speed"]
            else:
                move_speed = self.Speed
        else:
            move_speed = self.para["move_max_speed"]
        # distance = 30 * self.para["move_max_speed"]
        
        new_evade_pos = TSVector3.plus(self.pos3d, TSVector3.multscalar(new_dir, distance))
        if self.Z < 3000 or new_evade_pos['Z']<2000:
            new_evade_pos['Z'] = self.para["area_max_alt"]-500
        elif self.Z > self.para["area_max_alt"]-1000 or new_evade_pos['Z']>self.para["area_max_alt"]:
            new_evade_pos['Z'] = 3000
        vertical_evade_route_list = [{"X": new_evade_pos['X'], "Y": new_evade_pos['Y'], "Z": new_evade_pos['Z']}, ]
        if (abs(new_evade_pos["X"]) >= 142000 or abs(new_evade_pos["Y"]) >= 145000):
            cmd_list.append(env_cmd.make_areapatrolparam(self.ID, self.X, self.Y, vertical_evade_route_list[0]["Z"], move_speed, 100, self.para["move_max_speed"],
                                        self.para["move_max_acc"], self.para["move_max_g"]))
            
        if (abs(new_evade_pos["X"]) < 142000 and abs(new_evade_pos["Y"]) < 145000):
            cmd_list.append(env_cmd.make_linepatrolparam(self.ID, vertical_evade_route_list,
                                        move_speed,
                                        self.para["move_max_acc"], self.para["move_max_g"]))

    def can_attack(self, enemy, attack_dis=0):# 根据视野和攻击距离判断是否可以攻击
        if self.Availability == 0:
            return False
        if attack_dis==0:
            attack_dis = self.para['launch_range']
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        if dis < attack_dis and self.can_see(enemy, see_factor=1):
            return True
        return False

    def can_see(self, enemy, see_factor = 1.1, jam_dis=0):# 判断飞机是否能看到敌方实体
        if jam_dis == 0:
            jam_dis = self.para['radar_range']*see_factor
        if self.Availability == 0:
            return False
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        enemy_theta = self.XY2theta(enemy.X - self.X,enemy.Y-self.Y)
        vector = TSVector3.minus(self.pos3d, enemy.pos3d)
        relative_pitch = TSVector3.calpitch(vector)
        if dis <= jam_dis and \
            abs(self.pi_bound(enemy_theta - self.Heading))*180/math.pi <= self.para['radar_heading']*see_factor and \
            abs(self.pi_bound(relative_pitch))*180/math.pi <= self.para['radar_pitch']*see_factor:
            if dis>self.para['radar_range']*0.4:
                enemy.close_distance = dis
            return True
        return False

    def XY2theta(self, X, Y):# 计算XOY面两向量夹角
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

    def pi_bound(self, theta):# 纠正夹角范围
        while theta > math.pi:
            theta -= 2*math.pi
        while theta <= -math.pi:
            theta += 2*math.pi
        return theta

    def limit_height(self, angle, bottom_line):
        horizontal_dis = 2 * self.para["move_max_speed"]
        if angle == 0:
            return horizontal_dis, 0
        if abs(bottom_line - self.Z) < math.tan(angle)*horizontal_dis:
            return math.sqrt(horizontal_dis**2 + (bottom_line - self.Z)**2), math.atan2(abs(bottom_line - self.Z),horizontal_dis)
        else:
            return horizontal_dis*math.cos(angle), angle
    

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
        self.fire_time = missile_info['CurTime']
        # 导弹预计到达时间
        if missile_info['distance']>2700:
            self.arrive_time = self.fire_time + missile_info['distance']/1000 + 7
        else:
            self.arrive_time = self.fire_time + (math.sqrt(350**2+4*49*missile_info['distance'])-350)/98
        # 上一帧距离
        self.pre_dis = missile_info['distance']
        # 类型
        self.Type = missile_info['Type']
        # 是否过靶
        self.loss_target = False
        # 被袭击的平台编号
        self.EngageTargetID = missile_info['EngageTargetID']
        # 军别信息
        self.Identification = missile_info['Identification']
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.lost_flag = 0

    def update_agent_info(self, missile_info):
        missile_acc = missile_info['Speed'] - self.Speed
        if missile_acc == 0:
            missile_acc = 98
        if missile_info['distance'] - self.pre_dis>0:
            self.loss_target = True
        self.pre_dis = missile_info['distance']
        self.Speed = missile_info['Speed']
        self.X = missile_info['X']
        self.Y = missile_info['Y']
        self.Z = missile_info['Alt']
        self.Pitch = missile_info['Pitch']
        if self.Speed**2+2*missile_acc*missile_info['distance']>0:
            self.arrive_time = missile_info['CurTime'] + (math.sqrt(self.Speed**2+2*missile_acc*missile_info['distance'])-self.Speed)/missile_acc
        else:
            self.arrive_time = missile_info['CurTime'] + missile_info['distance']/missile_info['Speed']
        if missile_info['Heading'] < 0:
            missile_info['Heading'] += math.pi * 2
        self.Heading = missile_info['Heading']
        self.Roll = missile_info['Roll']
        self.LauncherID = missile_info['LauncherID']
        self.EngageTargetID = missile_info['EngageTargetID']
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.lost_flag = 0
 