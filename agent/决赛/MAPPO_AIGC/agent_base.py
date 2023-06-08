import numpy as np
from env.env_cmd import CmdEnv as env_cmd
import random
import math
from utils.utils_math import TSVector3

class Plane(object):
    def __init__(self, agent):
        ## 飞机基本属性信息
        # 平台名字
        self.Name = agent['Name']
        self.i = -1
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
        # 上一帧俯仰角
        self.prePitch = 0
        # 横滚角
        self.Roll = agent['Roll']
        # 航向, 即偏航角
        if agent['Heading'] < 0:
            agent['Heading'] += math.pi * 2
        self.Heading = agent['Heading']
        # 速度
        self.Speed = agent['Speed']
        self.pre_speed = agent['Speed']
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
        # 上一帧坐标
        self.prePos3d = None
        # 是否在执行干扰任务
        self.do_jam = False
        if self.Type == 1:
            # 剩余弹药
            self.AllWeapon = 4
            # 上一次执行干扰时间
            self.last_jam = 0
            # 就绪导弹
            self.ready_missile = 4
            self.plane_acc = 9.8
        else:
            self.AllWeapon = 2
            # 就绪导弹
            self.ready_missile = 2
            self.plane_acc = 9.8*2
        ## 我方飞机相关信息绑定
        # 占据中心时间
        self.center_time = 0
        # 是否丢失
        self.lost_flag = 0
        # 飞机属性参数
        self.para = {}
        # 是否有移动命令
        self.move_order = None
        # 已经被提供了干扰
        self.jammed = -100
        # 僚机
        self.wing_plane = None
        # 是否被可提供干扰的有人机看到
        self.be_seen_leader = False
        # 是否被可提供干扰的有人机看到
        self.be_seen_myleader = False
        # 是否可以被监视
        self.be_seen = False
        # 是否处在威胁之中
        self.be_in_danger = False
        # 已发射导弹
        self.used_missile_list = []
        
        ## 敌方飞机相关信息绑定
        # 最近的一枚导弹
        self.close_missile = None
        # 被几发导弹瞄准
        self.locked_missile_list = []
        # 已经被几发导弹攻击过
        self.lost_missile_list = []
        # 威胁系数
        self.threat_ratio = 0
        # 根据危险系数计算移动速度
        self.move_speed = 300

        # 平台具体属性信息
        if self.Type == 1:
            self.para["move_max_speed"] = 400
            self.para["move_min_speed"] = 150
            self.para["move_max_acc"] = 1
            self.para["move_max_g"] = 6
            self.para["area_max_alt"] = 15000
            self.para['radar_range'] = 80000
            self.para["launch_range"] = 15000
            self.para["safe_range"] = 25000
            self.para['radar_heading'] = 60
            self.para['radar_pitch'] = 60
        else:
            self.para["move_max_speed"] = 300
            self.para["move_min_speed"] = 100
            self.para["move_max_acc"] = 2
            self.para["move_max_g"] = 12
            self.para["area_max_alt"] = 10000
            self.para['radar_range'] = 60000
            self.para["safe_range"] = 16000
            self.para["launch_range"] = 15000
            self.para['radar_heading'] = 30
            self.para['radar_pitch'] = 10

    def update_agent_info(self, agent, info_type):
        self.X = agent['X']
        self.Y = agent['Y']
        self.Z = agent['Alt']
        self.prePitch = self.Pitch
        self.Pitch = agent['Pitch']
        if agent['Heading'] < 0:
            agent['Heading'] += math.pi * 2
        self.Heading = agent['Heading']
        self.Roll = agent['Roll']
        self.pre_speed = self.Speed
        self.Speed = agent['Speed']
        if self.Speed>self.para["move_max_speed"]:
            self.Speed = self.para["move_max_speed"]
        self.CurTime = agent['CurTime']
        self.Availability = agent['Availability']
        self.lost_flag = 0
        self.prePos3d = self.pos3d
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        if info_type==0:
            self.ready_missile = agent['LeftWeapon']
            self.plane_acc = agent['AccMag']*9.8
        else:
            self.plane_acc = self.Speed - self.pre_speed
            if self.Speed+self.plane_acc > self.para["move_max_speed"]:
                self.plane_acc = self.para["move_max_speed"] - self.Speed

    def imaginative_update_agent_info(self,missile):
        if missile.Speed<1000:
            ans_t = self.Speed/(missile.Speed+missile.missile_acc)
        else:
            ans_t = self.Speed/missile.Speed
        ans_t = missile.arrive_time-missile.CurTime-ans_t
        new_dir = TSVector3.calorientation(missile.Heading, missile.prePitch)
        next_imaginative_point = TSVector3.plus(missile.pos3d, TSVector3.multscalar(new_dir, ans_t*missile.Speed))
        self.Pitch = TSVector3.calpitch(TSVector3.minus(next_imaginative_point, self.pos3d))
        self.X = next_imaginative_point['X']
        self.Y = next_imaginative_point['Y']
        self.Z = next_imaginative_point['Z']
        self.pre_speed = self.Speed
        self.Speed = self.para["move_max_speed"]
        self.plane_acc = self.Speed - self.pre_speed
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}

    def can_attack(self, enemy, attack_dis=0):# 根据视野和攻击距离判断是否可以攻击
        if self.Availability == 0 or self.ready_missile==0:
            return False
        if attack_dis==0:
            attack_dis = self.para['launch_range']
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        if dis < attack_dis and self.can_see(enemy, see_factor=0.99):
            return True
        return False

    def can_see(self, enemy, see_factor = 1.01, jam_dis=0):# 判断飞机是否能看到敌方实体
        if jam_dis == 0:
            jam_dis = self.para['radar_range']*see_factor
        if self.Availability == 0:
            return False
        dis = TSVector3.distance(self.pos3d, enemy.pos3d)
        enemy_theta = self.XY2theta(enemy.X - self.X,enemy.Y-self.Y)
        vector = TSVector3.minus(self.pos3d, enemy.pos3d)
        relative_pitch = TSVector3.calpitch(vector)
        if dis < jam_dis and \
            abs(self.pi_bound(enemy_theta - self.Heading))*180/math.pi < self.para['radar_heading']*see_factor and \
            abs(self.pi_bound(relative_pitch))*180/math.pi < self.para['radar_pitch']*see_factor:
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
        # 上一帧俯仰角
        self.prePitch = self.Pitch
        # 航速
        self.Speed = missile_info['Speed']
        # 航向, 即偏航角
        if missile_info['Heading'] < 0:
            missile_info['Heading'] += math.pi * 2
        self.Heading = missile_info['Heading']
        # 上一帧朝向
        self.preHeading = self.Heading
        # 来袭导弹发射平台编号
        self.LauncherID = missile_info['LauncherID']
        # 仿真时间
        self.CurTime = missile_info['CurTime']
        # 发射时间
        self.fire_time = missile_info['CurTime']-1
        self.arrive_time = self.fire_time
        # 上一帧距离
        self.pre_dis = missile_info['distance']
        # 初始距离
        self.init_dis = missile_info['distance']
        # 上一帧加速度
        self.missile_acc = 98
        # 类型
        self.Type = missile_info['Type']
        # 是否过靶
        self.loss_target = False
        # 被袭击的平台编号
        self.EngageTargetID = missile_info['EngageTargetID']
        # 军别信息
        self.Identification = missile_info['Identification']
        self.prePos3d = None
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        # 是否丢失视野
        self.lost_flag = 0

    def update_agent_info(self, missile_info, info_type):
        self.missile_acc = missile_info['Speed'] - self.Speed
        if self.missile_acc + self.Speed>1000:
            self.missile_acc = 1000-self.Speed
        if missile_info['distance'] - self.pre_dis>0:
            self.loss_target = True
        self.pre_dis = missile_info['distance']
        self.Speed = missile_info['Speed']
        self.X = missile_info['X']
        self.Y = missile_info['Y']
        self.Z = missile_info['Alt']
        self.CurTime = missile_info['CurTime']
        self.prePitch = self.Pitch
        self.Pitch = missile_info['Pitch']
        if missile_info['Heading'] < 0:
            missile_info['Heading'] += math.pi * 2
        self.preHeading = self.Heading
        self.Heading = missile_info['Heading']
        self.Roll = missile_info['Roll']
        self.prePos3d = self.pos3d
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.lost_flag = 0
    
    def imaginative_update_agent_info(self,target_dis,target_heading,target_pitch):
        self.Speed = self.missile_acc + self.Speed
        if self.Speed > 1000:
            self.Speed = 1000
            self.missile_acc = 0
        if target_dis - self.pre_dis>0:
            self.loss_target = True
        self.pre_dis = target_dis
        new_dir = TSVector3.calorientation(self.Heading, self.Pitch)
        next_imaginative_point = TSVector3.plus(self.pos3d, TSVector3.multscalar(new_dir, self.Speed))
        self.X = next_imaginative_point['X']
        self.Y = next_imaginative_point['Y']
        self.Z = next_imaginative_point['Z']
        if target_heading < 0:
            target_heading += math.pi * 2
        elif target_heading>math.pi*2:
            target_heading -= math.pi * 2
        self.Pitch = target_pitch
        self.Heading = target_heading
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.lost_flag = 0
 