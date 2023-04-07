from env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
from agent.WK_AI_Defence_v2.utils import reg_rad, dir2rad, adjust_angle_to_target, check_dis
import numpy as np
import copy


class MyJet(object):
    def __init__(self, agent):
        self.Name = agent['Name']
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

        self.start_time, self.start_x, self.start_y = 0, 0, 0

        ## 初始化编队阵型
        self.init_battle_info()
        self.static_para()

        self.jam_free = True
        self.jam_using = False
        self.start_time = 0

    def __eq__(self, other):
        return True if self.ID == other.ID else False

    def init_battle_info(self):
        # 初始化jet所属的编队
        formation_1st = ["红有人机1", "红无人机1", "红无人机2", "红无人机3", "红无人机4",
                         "蓝有人机1", "蓝无人机1", "蓝无人机2", "蓝无人机3", "蓝无人机4"]
        formation_2nd = ["红有人机2", "红无人机5", "红无人机6", "红无人机7", "红无人机8",
                         "蓝有人机2", "蓝无人机5", "蓝无人机6", "蓝无人机7", "蓝无人机8"]
        if self.Name in formation_1st:
            self.formation = "first"
        if self.Name in formation_2nd:
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
            ##虽然能探测80 但是攻击距离是60？？
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
            ## 这几个距离怎么算出来的？？
            self.Raderdis = 39e3
            self.RadarHorizon = 30
            self.RadarVertical = 10

            self.AttackRadarDis = 45e3
            self.AttackRadarHorizon = 50
            self.AttackRadarVertical = 50
    
    def evade(self, sim_time, enemy, cmd_list):
        EntityPos = {"X": self.X, "Y": self.Y, "Z": self.Z}
        EnemyPos = {"X": enemy.X, "Y": enemy.Y, "Z": enemy.Z}

        dis = TSVector3.distance(EntityPos, EnemyPos)
        if EntityPos["Z"] >= 2000 and EntityPos["Z"] <= (self.max_alt + 2000) / 2.0:
            alt = 2000
        else:
            alt = self.max_alt

        EnemyPos["Z"] = EntityPos["Z"]
        vector = TSVector3.minus(EntityPos, EnemyPos)
        dir = TSVector3.normalize(vector)
        distance = 200 * self.max_speed
        evade_pos = TSVector3.plus(EntityPos, TSVector3.multscalar(dir, distance))
        straight_evade_route_list = [{"X": evade_pos["X"], "Y": evade_pos["Y"], "Z": EnemyPos["Z"]}, ]

        # 选择90度后的单位
        heading = TSVector3.calheading(dir)
        new_heading = heading + np.math.pi / 2
        new_dir = TSVector3.calorientation(new_heading, 0)
        new_evade_pos = TSVector3.plus(EntityPos, TSVector3.multscalar(new_dir, distance))
        #vertical_evade_route_list = [{"X": new_evade_pos['X'], "Y": new_evade_pos['Y'], "Z": alt}, ]
        vertical_evade_route_list = [{"X": new_evade_pos['X'], "Y": new_evade_pos['Y'], "Z": alt}, ]
        #print("敌我双方距离:",dis)

        '''
        ##效果并不明显
        if sim_time % 150 == 0:
            self.start_time = sim_time
            self.start_x, self.start_y = self.X, self.Y
            cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, self.Z, 200, 100, self.max_speed,
                                                self.max_acc, self.max_g))
        elif sim_time <= (self.start_time+40):
            #把盘旋动作做完
            cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.start_x, self.start_y, self.Z, 200, 100, self.max_speed,
                                                self.max_acc, self.max_g))
        '''

        '''
        if 1:
        #else:
            if dis > 20000:
                # 计算规避位置
                #print("要躲避位置:",evade_pos)
                #若计算后左右边界大于正常 内层再判断计算后的高度 执行区域巡逻
                #若计算后的左右边界正常范围内 内层再判断计算后的高度 则执行躲避
                if (evade_pos["Z"] > self.max_alt or evade_pos["Z"] < 2000):
                    straight_evade_route_list[0]["Z"] = alt
                if (abs(evade_pos["X"]) >= 142000 or abs(evade_pos["Y"]) >= 145000):
                    #print("我方{}ID为{}进行盘旋躲避".format(self.Name,self.ID))
                    cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, straight_evade_route_list[0]["Z"], 200, 100, self.max_speed,
                                                    self.max_acc, self.max_g))
                if (abs(evade_pos["X"]) < 142000 and abs(evade_pos["Y"]) < 145000):
                    #print("我方{}ID为{}进行反方向躲避".format(self.Name,self.ID))
                    cmd_list.append(CmdEnv.make_linepatrolparam(self.ID, straight_evade_route_list,
                                                self.max_speed,
                                                self.max_acc, self.max_g))

            else:
                #cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, self.Z, 100, 100, self.max_speed,
                #                                    self.max_acc, self.max_g))
                cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, self.Z, 100, 100, self.max_speed,
                                                    self.max_acc, self.max_g))
                #print("我方{}ID为{}进行盘旋躲避".format(self.Name,self.ID))
                '''

        if dis > 25000:
            # 计算规避位置
            #print("要躲避位置:",evade_pos)
            #若计算后左右边界大于正常 内层再判断计算后的高度 执行区域巡逻
            #若计算后的左右边界正常范围内 内层再判断计算后的高度 则执行躲避
            if (evade_pos["Z"] > self.max_alt or evade_pos["Z"] < 2000):
                straight_evade_route_list[0]["Z"] = alt
            if (abs(evade_pos["X"]) >= 142000 or abs(evade_pos["Y"]) >= 145000):
                #print("我方{}ID为{}进行盘旋躲避".format(self.Name,self.ID))
                cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, straight_evade_route_list[0]["Z"], 200, 100, self.max_speed,
                                                self.max_acc, self.max_g))
            if (abs(evade_pos["X"]) < 142000 and abs(evade_pos["Y"]) < 145000):
                #if self.Type == 1:
                #    print("我方{}ID为{}进行反方向躲避".format(self.Name,self.ID))
                cmd_list.append(CmdEnv.make_linepatrolparam(self.ID, straight_evade_route_list,
                                            self.max_speed,
                                            self.max_acc, self.max_g))

        else:
            if (new_evade_pos["Z"] >= self.max_alt or new_evade_pos["Z"] < 2000):
                vertical_evade_route_list[0]["Z"] = alt

            if (abs(new_evade_pos["X"]) >= 142000 or abs(new_evade_pos["Y"]) >= 145000):
                #if self.Type == 1:
                #    print("我方{}ID为{}进行盘旋躲避".format(self.Name,self.ID))
                cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.X, self.Y, vertical_evade_route_list[0]["Z"], 200, 100, self.max_speed,
                                                self.max_acc, self.max_g))
            if (abs(new_evade_pos["X"]) < 142000 and abs(new_evade_pos["Y"]) < 145000):
                #if self.Type == 1:
                #    print("我方{}ID为{}进行垂直躲避".format(self.Name,self.ID))
                cmd_list.append(CmdEnv.make_linepatrolparam(self.ID, vertical_evade_route_list,
                                            self.max_speed,
                                            self.max_acc, self.max_g))
                """
                if sim_time > self.start_time + 40:
                    self.start_time = sim_time
                    self.start_x, self.start_y = self.X, self.Y
                    print("--------sim_time:{}----".format(sim_time))
                cmd_list.append(CmdEnv.make_areapatrolparam(self.ID, self.start_x, self.start_y, vertical_evade_route_list[0]["Z"], 200, 100, self.max_speed,
                                                self.max_acc, self.max_g))
                """



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

        ## 发射的导弹列表
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
        self.Name = missile_info['Name']
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
        self.Name = missile_info['Name']
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