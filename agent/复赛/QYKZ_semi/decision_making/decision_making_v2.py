import copy
import numpy as np
from env.env_cmd import CmdEnv
import json
from agent.QYKZ_semi.algorithm.dbscan import DBSCAN
from utils.utils_math import TSVector3
import math
import random

class QYKZ_Decision_v2():
    def __init__(self, global_observation):
        self.global_observation = global_observation
        
        self._init_info()

    def _init_info(self):
        # 我方所有飞机信息列表
        self.my_jets = []
        # 我方有人机信息列表
        self.my_leader_jets = []
        # 我方无人机信息列表
        self.my_uav_jets = []

        # 敌方所有飞机信息列表
        self.enemy_jets = []
        # 敌方有人机信息列表
        self.enemy_leader_jets = []
        # 敌方无人机信息列表
        self.enemy_uav_jets = []

        # 我方导弹信息
        self.my_missile = []
        # 敌方导弹信息
        self.enemy_missile = []
        # 整体导弹信息
        self.total_missile_list = []

        # 被打击的敌机列表
        self.hit_enemy_list = []

        # 识别红军还是蓝军
        self.side = None

        self.cmd_list = []

        # 有人机第一编队
        self.first_leader_formation = {}
        # 有人机第二编队
        self.sec_leader_formation = {}

        # 无人机第一编队
        self.first_uav_formation = [0, 0]
        # 无人机第二编队
        self.sec_uav_formation = [0, 0]

        # 第一总编队
        self.first_formation = {}
        # 第二总编队
        self.sec_formation = {}

        # 躲避列表
        self.evade_list = []

        ## 编队的攻击角度
        self.Fleet_Attack_Angle = 110
        ## 编队的攻击距离
        self.Fleet_Attack_dis = 60000
        self.safe_dis = 80000
        ## 协同作战
        self.cooperative_combat_dic = None

        self.first_formation_init_target = {}
        self.sec_formation_init_target = {}

        self.DBSCAN_eps = 25000
        self.DBSCAN_min_sample = 2


    def reset(self):
        self._init_info()

    def update_entity_info(self):
        # 己方所有飞机信息
        self.my_jets = self.global_observation.observation.get_all_agent_list()
        # 己方有人机信息
        self.my_leader_jets = self.global_observation.observation.get_agent_info_by_type(1)
        # 己方无人机信息
        self.my_uav_jets = self.global_observation.observation.get_agent_info_by_type(2)
        # 敌方所有飞机信息
        self.enemy_jets = self.global_observation.perception_observation.get_all_agent_list()
        # 敌方有人机信息
        self.enemy_leader_jets = self.global_observation.perception_observation.get_agent_info_by_type(1)
        # 敌方无人机信息
        self.enemy_uav_jets = self.global_observation.perception_observation.get_agent_info_by_type(2)

        self.my_jets_dic = self.global_observation.observation.get_all_agent_dic()

        if len(self.enemy_leader_jets):
            self.bound = 40000 
        else:
            self.bound = 90000

        # 获取队伍标识
        if self.side is None:
            if self.my_jets[0].Identification == "红方":
                self.side = 1
            else:
                self.side = -1

        # 更新双方导弹信息
        step_missile_list = self.global_observation.missile_observation.get_missile_list()
        enemy_missile = []
        my_missile = []
        for missile in step_missile_list:
            if missile.Identification == self.my_jets[0].Identification:
                if missile.IsLost == 0:
                    my_missile.append(missile)
            else:
                if missile.IsLost == 0:
                    enemy_missile.append(missile)
            '''
            这类目标无外乎就是找到我方发射的导弹和攻击我方的导弹 这里实现的不对
            先注释掉 应该在导弹发射出来的那一刻进行添加

            ## 筛选还存活的missile 为啥ID不能是0呢
            missile_active = (missile.ID != 0 and missile.Availability > 0.0001)
            if not missile_active:
                continue
            ## 得到该枚导弹的发射方和攻击方
            launch_id = missile.LauncherID
            target_id = missile.EngageTargetID
            launch_jet = self.find_jet_by_given(launch_id, True)
            target_jet = self.find_jet_by_given(target_id, True)
            if launch_jet is not None:
                ##发射方飞机已发射列表进行添加
                ##是否会有重复添加的问题？？ 因为每步都在查询
                launch_jet.my_launch_missile.append(missile)
            if target_jet is not None:
                ##被攻击方的锁定导弹列表更新
                ##是否会有重复添加的问题？？ 因为每步都在查询
                target_jet.attacked_missile_list.append(missile)

            ##解决重复添加的问题
            '''
        self.total_missile_list = step_missile_list
        self.enemy_missile = enemy_missile
        self.my_missile = my_missile
        #update extra info
        for enemy_jet in self.enemy_jets:
            if enemy_jet.IsLost == 0:
                enemy_jet.Availability = 1
            else:
                #视野丢失 判断敌机是否还存活
                if len(enemy_jet.locked_missile_list) > 0 and enemy_jet.detect_dis != -1:
                    dead_flag = 0
                    for missile_id in enemy_jet.locked_missile_list:
                        missile = self.find_missile_by_given(missile_id, True)
                        if missile.IsLost:
                            dead_flag = 1
                            break
                    if dead_flag:
                        enemy_jet.Availability = 0
            for missile in self.total_missile_list:
                if missile.LauncherID == enemy_jet.ID and missile.ID not in enemy_jet.launched_missile_list:
                    enemy_jet.launched_missile_list.append(missile.ID)
                if (not missile.IsLost) and missile.EngageTargetID == enemy_jet.ID and missile.ID not in enemy_jet.locked_missile_list:
                    enemy_jet.locked_missile_list.append(missile.ID)
            for missile_id in enemy_jet.locked_missile_list.copy():
                miss_missile = self.find_missile_by_given(missile_id, True)
                if miss_missile is not None and miss_missile.IsLost:
                    enemy_jet.locked_missile_list.remove(missile_id)

        for my_jet in self.my_jets:
            if my_jet.IsLost:
                my_jet.Availability = 0
            for missile in self.total_missile_list:
                if missile.EngageTargetID == my_jet.ID and missile.ID not in my_jet.locked_missile_list and missile.IsLost==0:
                    #瞄准我方飞机的导弹
                    my_jet.locked_missile_list.append(missile.ID)
                if missile.LauncherID == my_jet.ID and missile.ID not in my_jet.launched_missile_list:
                    #已发射的导弹
                    my_jet.launched_missile_list.append(missile.ID)
            #本局还可用的导弹数 避免指令重复
            my_jet.step_valid_missile = my_jet.LeftWeapon -len(my_jet.launched_missile_list)
        

            




    def find_jet_by_given(self, name_or_id, id=False):
        ##根据name或者id找到jet
        for jet in self.my_jets + self.enemy_jets:
            if id == True:
                if jet.ID == name_or_id:
                    return jet
            else:
                if jet.Name == name_or_id:
                    return jet
        return None
    
    def find_missile_by_given(self, name_or_id, id=False):
        ##根据name或者id找到jet
        for missile in self.total_missile_list:
            if id == True:
                if missile.ID == name_or_id:
                    return missile
            else:
                if missile.Name == name_or_id:
                    return missile
        return None

    def formation(self):
        self.first_leader_formation["up_plane"] = self.my_uav_jets[0]
        self.first_leader_formation["down_plane"] = self.my_uav_jets[1]
        self.first_leader_formation["leader"] = self.my_leader_jets[0]
        self.sec_leader_formation["up_plane"] = self.my_uav_jets[2]
        self.sec_leader_formation["down_plane"] = self.my_uav_jets[3]
        self.sec_leader_formation["leader"] = self.my_leader_jets[1]

        self.first_uav_formation[0] = self.my_uav_jets[4]
        self.first_uav_formation[1] = self.my_uav_jets[5]

        self.sec_uav_formation[0] = self.my_uav_jets[6]
        self.sec_uav_formation[1] = self.my_uav_jets[7]

        self.first_formation["up_plane"] = self.my_uav_jets[0]
        self.first_formation["down_plane"] = self.my_uav_jets[1]
        self.first_formation["leader_plane"] = self.my_leader_jets[0]
        self.first_formation["uav_1"] = self.my_uav_jets[4]
        self.first_formation["uav_2"] = self.my_uav_jets[5]

        self.sec_formation["up_plane"] = self.my_uav_jets[2]
        self.sec_formation["down_plane"] = self.my_uav_jets[3]
        self.sec_formation["leader_plane"] = self.my_leader_jets[1]
        self.sec_formation["uav_1"] = self.my_uav_jets[6]
        self.sec_formation["uav_2"] = self.my_uav_jets[7]

        # 干扰
        self.jam_list = [[plane, 0] for plane in self.my_leader_jets]
    
    def init_pos(self, cmd_list):
        self.formation()
        # 初始化部署
        self.init_direction = 90
        if self.side == -1:
            self.init_direction = 270
        leader_plane_1 = self.first_leader_formation["leader"]
        leader_plane_2 = self.sec_leader_formation["leader"]
        # 初始化有人机位置
        cmd_list.append(
            CmdEnv.make_entityinitinfo(leader_plane_1.ID, -135000 * self.side, 75000, 9500, 180, self.init_direction))
        cmd_list.append(
            CmdEnv.make_entityinitinfo(leader_plane_2.ID, -135000 * self.side, -75000, 9500, 180, self.init_direction))

        for key, value in self.first_leader_formation.items():
            if key == "up_plane":
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(value.ID, -125000 * self.side, 85000, 9500, 180, self.init_direction))
            if key == "down_plane":
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(value.ID, -125000 * self.side, 65000, 9500, 180, self.init_direction))

        for key, value in self.sec_leader_formation.items():
            if key == "up_plane":
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(value.ID, -125000 * self.side, -65000, 9500, 180, self.init_direction))
            if key == "down_plane":
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(value.ID, -125000 * self.side, -85000, 9500, 180, self.init_direction))

        for i, plane in enumerate(self.first_uav_formation):
            cmd_list.append(
                CmdEnv.make_entityinitinfo(plane.ID, -140000 * self.side, 65000 - ((i + 1) * 10000), 9500, 180,
                                            self.init_direction))

        for i, plane in enumerate(self.sec_uav_formation):
            cmd_list.append(
                CmdEnv.make_entityinitinfo(plane.ID, -140000 * self.side, -65000 + ((i + 1) * 10000), 9500, 180,
                                            self.init_direction))
        
        #为有人机指定僚机
        #蓝有人机1的僚机：蓝无人机1 蓝无人机3
        #蓝有人机2的僚机：蓝无人机5 蓝无人机7
        leader_plane_1.formation_mate = [self.first_formation["up_plane"], self.first_formation["down_plane"]]
        leader_plane_2.formation_mate = [self.sec_formation["up_plane"], self.sec_formation["down_plane"]]


    def init_target(self):
        '''
        前三秒 指定初始目标
        '''
        
        first_formation_route = {"up_plane": {"X": 40000 * self.side, "Y": 85000, "Z": 9500},
                                         "down_plane": {"X": 40000 * self.side, "Y": 65000, "Z": 9500},
                                         "leader_plane": {"X": 145000 * self.side, "Y": 75000, "Z": 9500},
                                         "uav_1": {"X": 30000 * self.side, "Y": 55000, "Z": 9500},
                                         "uav_2": {"X": 30000 * self.side, "Y": 45000, "Z": 9500}
                                         }
        sec_formation_route = {"up_plane": {"X": 40000 * -self.side, "Y": -65000, "Z": 9500},
                                "down_plane": {"X": 40000 * -self.side, "Y": -85000, "Z": 9500},
                                "leader_plane": {"X": 145000 * -self.side, "Y": -75000, "Z": 9500},
                                "uav_1": {"X": 30000 * -self.side, "Y": -55000, "Z": 9500},
                                "uav_2": {"X": 30000 * -self.side, "Y": -45000, "Z": 9500}
                                }
        
        for plane, route in first_formation_route.items():
            init_target = copy.deepcopy(self.my_jets[0])
            init_target.X, init_target.Y, init_target.Z = route["X"], route["Y"], route["Z"]
            init_target.pos2d = np.array([init_target.X, init_target.Y], dtype=float)
            init_target.pos3d = np.array([init_target.X, init_target.Y, init_target.Z],dtype=float)
            self.first_formation[plane].target = init_target
            self.first_formation_init_target[plane] = init_target
        for plane, route in sec_formation_route.items():
            init_target = copy.deepcopy(self.my_jets[0])
            init_target.X, init_target.Y, init_target.Z = route["X"], route["Y"], route["Z"]
            init_target.pos2d = np.array([init_target.X, init_target.Y], dtype=float)
            init_target.pos3d = np.array([init_target.X, init_target.Y, init_target.Z],dtype=float)
            self.sec_formation[plane].target = init_target
            self.sec_formation_init_target[plane] = init_target

    def jet_pos_cluster(self, eps, MinPts, side):
        '''
        对探测到的敌人/我方飞机进行聚类分析
        若探测到敌机<=2个 直接返回两类
        若探测到敌机>2个 进行聚类分析
        return jet_pos_list
        jet_pos_list是由list组成的
        每个list中由同属一个cluster的敌/我机组成
        同一个cluster中的敌/我机，在25Km内都可以找到邻居
        方圆25KM内没有邻居的敌/我机独自成cluster
        '''
        if side == 'enemy':
            pos_list = [[jet.X, jet.Y] for jet in self.enemy_jets if jet.Availability]
        elif side == 'my':
            pos_list = [[jet.X, jet.Y] for jet in self.my_jets if jet.Availability]
        jet_pos_cluster = DBSCAN(pos_list, eps, MinPts)
        ## eps:25KM min_sample:2
        ##同属一个cluster的敌/我机 相邻敌/我机之间的距离都在25KM以内 label都是相同的
        ##单机的label是-1 最小的cluster内有两架
        ##enemy_pos_cluster存储的就是各个敌机的label

        labels = []
        jet_pos_list = []
        if side == "enemy":
            exist_jets = self.enemy_jets
        elif side == "my":
            exist_jets = self.my_jets

        
    
        for pos_c, jet in enumerate(exist_jets):
            label = jet_pos_cluster[pos_c]
            if label not in labels and label != -1:
                #先不处理单独成类的
                labels.append(label)
                jet_pos_list.append([])
                # jet_pos_list也由list组成 每个list中的敌/我机同属一个cluster
                jet_pos_list[label - 1].append(jet)
            elif label in labels:
                jet_pos_list[label - 1].append(jet)
        #处理单独成类的
        for pos_c, jet in enumerate(exist_jets):
            label = jet_pos_cluster[pos_c]
            if label == -1:
                jet_pos_list.append([jet])

        return jet_pos_list

    def get_nearest_jet(self, jets_list1, jets_list2, flag=False):
        """
        从两个战机集合中，依据距离找到最近的一对最近的战机

        Args:
            jets_list1:敌机cluster
            jets_list2:我方飞机

        Returns:
            两个战机实体

        """
        min_dis = 99999999
        attack_jet = None
        attacked_jet = None
        for enemy_jet in jets_list1:
            for my_jet in jets_list2:
                dis = TSVector3.distance(my_jet.pos3d, enemy_jet.pos3d)
                if dis < min_dis:
                    min_dis = dis
                    attack_jet = my_jet
                    attacked_jet = enemy_jet
        if attack_jet and attacked_jet:
            if flag:
                return attack_jet, attacked_jet, min_dis
            else:
                return attack_jet, attacked_jet
        else:
            return None, None

    def update_track_target(self):
        '''
        初始朝假目标移动
        有目标后进行追踪
        只是更新要跟踪的目标 并不下达具体的指令
        '''
        if len(self.enemy_jets) == 0:
            # print("未探测到敌人，朝中心点移动")
            for formation, jet in self.first_formation.items():
                jet.target = self.first_formation_init_target[formation]
            for formation, jet in self.sec_formation.items():
                jet.target = self.sec_formation_init_target[formation]
        else:
            ##若探测到敌方 先给敌方聚个类 然后判断去哪一拨
            ##问题：如果探测到好几波 是不是要分散兵力
            ##现在实现的是就近原则
            ##思路是为每个编队找target 而不是为了每个cluster找攻击的飞机
            #优点是 同一个cluster可能会有多个编队攻击 而且不用考虑编队内目标冲突
            ##还存在的问题是容易产生饱和式攻击 比如我方10架飞机全都聚成一类了 

            # 依据敌人之间互相的位置对敌人进行聚类
            # 返回的list也是由list组成的
            # 每个list中的敌机同属一个cluster
            eps = self.DBSCAN_eps
            MinPts = self.DBSCAN_min_sample
            enemy_pos_list = self.jet_pos_cluster(eps, MinPts, side="enemy")
            #print("当前探测到敌机共{}架，聚为{}类".format(len(self.enemy_jets), len(enemy_pos_list)))
            #for i, cluster in enumerate(enemy_pos_list):
            #    print("第{}个类，此类中有敌机：{}".format(i, [enemy.Name for enemy in cluster]))
            my_pos_list = self.jet_pos_cluster(eps, MinPts, side="my")
            #print("我方飞机现存{}架，聚为{}类".format(len(self.my_jets), len(my_pos_list)))
            #for i, cluster in enumerate(my_pos_list):
            #    print("第{}个类，此类中有我机：{}".format(i, [enemy.Name for enemy in cluster]))
            for my_formation in my_pos_list:
                min_dis = 9999999
                attack_jet = None
                target = None 
                for enemy_formation in enemy_pos_list:
                    temp_attack_jet, temp_target, temp_dis = self.get_nearest_jet(enemy_formation,my_formation,  flag=True)
                    if temp_dis < min_dis:
                        min_dis = temp_dis
                        attack_jet = temp_attack_jet
                        target = temp_target

                for jet in my_formation:
                    #print("我方飞机{}的目标为{}".format(jet.Name, target.Name))
                    jet.target = target
            
            '''
            #for cluster in enemy_pos_list:
            #    attack_jet, target = self.get_nearest_jet(cluster, self.my_jets)
            #思路是为每个编队找target 而不是为了每个cluster找攻击的飞机
            #优点是 同一个cluster可能会有多个编队攻击 而且不用考虑编队内目标冲突
            for formation in self.formation_list:
                ##遍历我方每个编队 为每个编队指派目标cluster
                ##若当前target死亡 ，编队的target会不会继续更新？？
                ##不用担心更新的问题 因为死亡以后self.my_jets/self.enemy_jets会自动更新
                #由于现在多个飞机会指向同一个目标 要注意饱和式攻击的问题
                print("当前我方编队为：{}".format([jet.Name for jet in formation]))
                for cluster in enemy_pos_list:
                    attack_jet, target = self.get_nearest_jet(cluster,formation)
                    ##formation可以覆盖所有的我方飞机 不用担心target冲突
                    ##可能多个编队的target都属于同一个cluster
                    ##敌机是否是有人机 挂弹量如何暂时没有考虑
                    for jet in formation:
                        jet.target = target 
                print("当前编队的目标为：",target.Name)
            '''

    def formation_track(self, jet):
        """
        列队呈包夹队形朝目标飞行
        """
        # 得到此列队的飞机数目 2/3
        member_num = len(jet.formation_mate) + 1
        # 得到jet与target之间的delta
        delta = jet.pos2d - jet.target.pos2d
        # 得到编组内的其他成员
        # 不一定是uav啊 有人机也可能无人机的mate
        uav_mates = [self.find_jet_by_given(mate) for mate in jet.formation_mate]
        # 得到编组内所有成员与target之间的delta
        delta_all = [mate.pos2d - jet.target.pos2d for mate in uav_mates]
        delta_all.append(delta)
        # 获得当前战机的小队中心位置坐标
        squad_center = jet.pos2d / member_num
        for mate in uav_mates:
            squad_center += mate.pos2d / member_num
        # 计算小队成员的弧度差排序
        ##这段代码没有搞明白什么意思
        rad_basement = dir2rad(squad_center - jet.target.pos2d)
        rad_all = [reg_rad_at(dir2rad(d) - rad_basement, 0)
                   for d in delta_all]
        rank_index = np.argsort(rad_all)
        rank_at = np.where(rank_index == (member_num - 1))[0][0]

        # 根据当前飞机在小队中的位置，计算目标点
        if member_num == 1:
            # 若小队成员只有自己，那么直接朝目标敌人飞去
            dest_now3d = jet.target.pos3d
        else:
            ##这段目前也完全没看懂
            # 110*(pi/180)
            attack_rad = self.Fleet_Attack_Angle * (np.pi / 180)
            attack_rad = self.Fleet_Attack_Angle * (np.pi / 180)
            jet_rad = rad_basement - attack_rad / 2 + rank_at * (attack_rad / (member_num - 1))
            vec2d = self.Fleet_Attack_dis * np.array([np.cos(jet_rad), np.sin(jet_rad)])
            dest_now3d = copy.deepcopy(jet.target.pos3d)
            dest_now3d[:2] += vec2d

        ##这段目前也完全没看懂
        if jet.Type == 1:
            dest_now2d = dest_now3d[:2]
            dest_now2d = self.shrink_2d_vec(jet, dest_now2d, dis=10e3)
            dest_location = [{"X": dest_now2d[0], "Y": dest_now2d[1], "Z": jet.max_alt}]
        else:
            dest_now2d = dest_now3d[:2]
            dest_now2d = self.shrink_2d_vec(jet, dest_now2d, dis=20e3)
            dest_location = [{"X": dest_now2d[0], "Y": dest_now2d[1], "Z": 8500}]

            get_enemy_jet_avg_height = [enemy_jet.Z for enemy_jet in self.enemy_jets if enemy_jet.Type == 2]
            if len(get_enemy_jet_avg_height) > 0:
                avg_height = np.array(get_enemy_jet_avg_height).mean()
                dest_location = [{"X": dest_now2d[0], "Y": dest_now2d[1], "Z": avg_height}]

        # 调整行径速度
        jet_dis = get_dis(jet, jet.target)
        squad_target_distance = [get_dis(
            mate, jet.target) for mate in uav_mates]
        squad_target_distance.append(jet_dis)

        dis_farthest = [get_dis(mate, jet.target) for mate in uav_mates
                        if mate.Type == 2]
        if jet.Type == 2:
            dis_farthest.append(jet_dis)
        baseline_dis = max(dis_farthest)

        ideal_dis = baseline_dis if jet.Type == 2 else baseline_dis + 15e3
        dis_error = jet_dis - ideal_dis

        speed = self.thresh_hold_projection(
            dis_error, min_x=-3e3, y_min_x=jet.max_speed * 0.5,
            max_x=0, y_max_x=jet.max_speed
        )
        if jet.Type == 1:
            speed = self.thresh_hold_projection(dis_error,
                                                min_x=-4e3, y_min_x=200,
                                                max_x=4e3, y_max_x=400)
        self.cmd_list.append(check_and_make_linepatrolparam(
            jet,
            dest_location,
            speed,
            jet.max_acc,
            jet.max_g
        ))

    def shrink_2d_vec(self, p, dest, dis):
        vec2dst = dest[:2] - p.pos2d
        dst_now2d = vec2dst / (np.linalg.norm(vec2dst) + 1e-7) * dis
        return p.pos2d + dst_now2d

    @staticmethod
    def thresh_hold_projection(x, min_x, y_min_x, max_x, y_max_x):
        assert min_x < max_x

        if x <= min_x:
            return y_min_x
        if x >= max_x:
            return y_max_x

        # 处于线性区间内
        k = (y_max_x - y_min_x) / (max_x - min_x)
        dx = x - min_x
        return dx * k + y_min_x

    def tactical_track(self, jet):
        ##根据给每个战机分配的target 更新命令
        # 首先找到当前jet的队友
        #jet_mates = [self.find_jet_by_given(name) for name in jet.formation_mate]
        # 过滤掉已经阵亡的队友
        #jet_mates = [mate for mate in jet_mates if mate is not None]
        # 更新当前jet的编队队友列表
        #jet.formation_mate = [mate.Name for mate in jet_mates]
        ##已对上述代码逻辑进行了改进 对己方也会进行聚类

        # 如果没有敌人则把初始目标点当作敌人, 进行未接敌的组队移动
        # init_move用于判断是否要执行朝向假目标飞行

        evade_plane_list = [plane["plane_entity"] for plane in self.evade_list]
        init_move = len(self.enemy_jets) == 0
        attack_dis = get_dis(jet, jet.target)

        if (init_move or attack_dis > self.Fleet_Attack_dis) and (jet not in evade_plane_list):
            # 若没有发现敌机或者距离目标点60KM以上
            # 进行编组移动（包夹队形）
            # print("未发现敌机或者距离目标点60KM以上，呈包夹队形移动")
            #self.formation_track(jet)
            ##包夹队形暂时舍弃 用最简单的
            dest_location = [{"X": jet.target.X, "Y": jet.target.Y, "Z": jet.max_alt * 0.9}]
            self.cmd_list.append(check_and_make_linepatrolparam(
            jet,
            dest_location,
            200,
            jet.max_acc,
            jet.max_g
            ))

        elif attack_dis < self.Fleet_Attack_dis:
            # 敌机进入攻击距离内 60KM
            #print("敌机进入攻击范围")
            pass
            ##这里需要根据当前态势拟定不同的作战方案
            ##特殊情况下：我方飞机为无人机目标为有人机 且没弹了 直接去朝着敌机有人机飞 进行制导

            

            '''
            if jet.LeftWeapon == 0 and jet.Type == 2 and jet.target.Type == 1 and get_dis(jet, jet.target) < 14e3:
                dest_loaction = jet.close_target_tracking(jet.target)
                self.cmd_list.append(check_and_make_linepatrolparam(
                    jet,
                    dest_loaction,
                    jet.max_speed,
                    jet.max_acc,
                    jet.max_g
                ))
            else:
                # 无特殊情况，有人机调整速度朝目标大致位置飞行， 无人机直接跟随目标飞行
                if jet.Type == 1:
                    _, threat_distance = jet.get_nearest_threat(self.enemy_jets)
                    if threat_distance < 30e3:
                        speed = 400
                    elif threat_distance < 40e3:
                        speed = 350
                    else:
                        speed = 300

                    dest_location = [{
                        "X": jet.target.X, "Y": jet.target.Y, "Z": jet.max_alt * 0.9
                    }]
                    self.cmd_list.append(check_and_make_linepatrolparam(
                        jet,
                        dest_location,
                        speed,
                        jet.max_acc,
                        jet.max_g
                    ))

                else:
                    self.cmd_list.append(
                        CmdEnv.make_followparam(
                            jet.ID,
                            jet.target.ID,
                            jet.max_speed,
                            jet.max_acc,
                            jet.max_g))
            '''
    def protect_leader_plane(self, sim_time):
        #在对局开始的前x秒内
        #只要视野内的敌有人机距我方有人挤60km内 就进行躲避
        if sim_time < 999999:
            for enemy in self.enemy_jets:
                if enemy.Type == 1:
                    for leader_plane in self.my_leader_jets:
                        temp_dis = get_dis(leader_plane, enemy)
                        if temp_dis < self.safe_dis:
                            evade_comb = {}
                            evade_comb["plane_entity"] = leader_plane
                            evade_comb["missile_entity"] = None
                            evade_comb['distance'] = temp_dis
                            self.evade_list.append(evade_comb)


    def update_evade(self):
        ##查看自身是否被导弹锁定
        #遍历敌方导弹 及其目标 让目标做出躲避动作
        #该列表由字典组成，每个字典有两个key:受攻击的我方飞机，攻击我方飞机的敌机导弹
        evade_id = [comb["missile_entity"].ID for comb in self.evade_list if comb["missile_entity"] is not None]


        # 统计所有被敌方导弹瞄准的我方飞机
        ##目前存在的问题：有可能一架飞机被多个导弹追
        if len(self.enemy_missile) != 0:
            for missile in self.enemy_missile:
                #print("导弹：{}，坐标x：{}，坐标y：{},坐标z:{}".format(missile.Name, missile.X, missile.Y, missile.Z))
                attacked_plane = self.global_observation.observation.get_agent_info_by_id(missile.EngageTargetID)
                if attacked_plane.Availability == 0:
                    continue
                if missile.ID not in evade_id:
                    evade_comb = {}
                    evade_comb["plane_entity"] = attacked_plane
                    evade_comb["missile_entity"] = missile
                    evade_comb['distance'] = TSVector3.distance(attacked_plane.pos3d, missile.pos3d)
                    self.evade_list.append(evade_comb)
 
            # 给危险程度分类 目前主要按照两者之间距离 还应该考虑飞机类型以及锁定导弹数量
            # dis_dict = {}
            # for comb in self.evade_list:
            #     if comb['distance'] not in dis_dict:
            #         dis_dict[comb['distance']] = comb["plane_entity"]
            #     else:
            #         dis_dict[comb['distance']+0.1] = comb["plane_entity"]
            # dange_evade_list = [value for key, value in sorted(dis_dict.items(), key=lambda d: d[0])]


        

        #判断危险是否解除
        #若危险解除则移除躲避列表
        for evade_plane in self.evade_list.copy():
            attacked_missile = evade_plane["missile_entity"]
            attacked_plane = evade_plane["plane_entity"]
            attacked_dis = evade_plane['distance']
            if attacked_missile is not None:
                missile_gone, over_target, safe_distance = False, False, False
                # 导弹已爆炸
                #if not self.global_observation.missile_observation.get_missile_info_by_rocket_id(attacked_missile.ID):
                if attacked_missile.IsLost:
                    missile_gone = True
                    #print("导弹：{}已爆炸".format(attacked_missile.Name))
                # 过靶
                missile_vector_3d = TSVector3.calorientation(attacked_missile.Heading, attacked_missile.Pitch)
                missile_vector = np.array([missile_vector_3d["X"], missile_vector_3d["Y"]])
                attacked_plane_pos = {"X": attacked_plane.X, "Y": attacked_plane.Y, "Z": attacked_plane.Z}
                attacked_missile_pos = {"X": attacked_missile.X, "Y": attacked_missile.Y, "Z": attacked_missile.Z}
                missile_mp_vector_3d = TSVector3.minus(attacked_plane_pos, attacked_missile_pos)
                missile_mp_vector = np.array([missile_mp_vector_3d["X"], missile_mp_vector_3d["Y"]])
                res = np.dot(np.array(missile_vector), np.array(missile_mp_vector)) / (
                        np.sqrt(np.sum(np.square(np.array(missile_vector)))) + 1e-9) / (
                              np.sqrt(np.sum(np.square(np.array(missile_mp_vector)))) + 1e-9)
                if abs(res) > 1:
                    res = res / abs(res)
                dir = np.math.acos(res) * 180 / np.math.pi
                if abs(dir) > 90:
                    over_target = True
                    #print("导弹：{}已过靶".format(attacked_missile.Name))
                # 飞到安全距离
                distance = TSVector3.distance(attacked_missile_pos, attacked_plane_pos)
                #if distance >= 100000:
                #    safe_distance = True
                ##满足任何一种情况：已爆炸、过靶、飞到安全距离 都移除 即不需要再躲避
                if any([missile_gone, over_target, safe_distance]):
                #if any([missile_gone, over_target]) and safe_distance:
                    self.evade_list.remove(evade_plane)

                ##躲避的时候为了防止被多个导弹追击 侦察不到 想了几个方案：
                ##1.躲避途中 自转一圈
                ##2.同属一个cluster的没有躲避任务的飞机提供一下支持
                ##3.加干扰
            else:
                pass
                #受保护的有人挤 移出躲避列表的条件是视野内没有敌方有人机
                #这块其实也有问题 若敌方有人挤无弹 或者需要攻击时
                #if attacked_plane.Type == 1:
                #    enemy_leader = False
                #    for enemy in self.enemy_jets:
                #        if enemy.Type == 1:
                #            if get_dis(enemy, attacked_plane) < self.safe_dis:
                #                enemy_leader = True
                #    if not enemy_leader:
                #        #危险解除
                #        self.evade_list.remove(evade_plane)



        #self.evade_list = evade_list
    
    def update_jam(self, sim_time):
        """
        只有 有人机 可以进行干扰
        也只为有人机进行干扰
        遍历有人机：
        若有人机需要躲避 使用自身以及不在躲避列表的另一个有人机使用干扰 前提条件是过了冷却时间 
        若是自身需要躲避 直接打开干扰
        若是支援别机 直接pass吧 改航向会有危险
        """
        evade_jets = [plane["plane_entity"] for plane in self.evade_list]
        for jet in self.my_leader_jets:
            #遍历有人机
            if jet in evade_jets:
                #处于躲避列表
                if jet.jam_free and (not jet.jam_using):
                    #可使用干扰状态 第一秒开启
                    jet.jam_using = True
                    jet.jam_free = False
                    jet.start_time = sim_time
                    self.cmd_list.append(CmdEnv.make_jamparam(jet.ID))
                
            if jet.jam_using:
                #只会持续3秒
                if sim_time <= (jet.start_time + 3):
                    self.cmd_list.append(CmdEnv.make_jamparam(jet.ID))
                else:
                    jet.jam_using = False 

            if sim_time > (jet.start_time + 3 + 60):
                #可以重新使用制导功能了
                jet.jam_free = True

        """
        if sim_time % 61 == 0:
            self.start_time = sim_time
        if sim_time <= (self.start_time + 3):
            #能够执行干扰的飞机
            jam_jets = [jet for jet in self.my_leader_jets]
            #需要躲避的飞机
            evade_jets = [plane["plane_entity"] for plane in self.evade_list]
            dis_dict = {}
            for comb in self.evade_list:
                if comb['distance'] not in dis_dict:
                    dis_dict[comb['distance']] = comb["plane_entity"]
                else:
                    dis_dict[comb['distance']+0.1] = comb["plane_entity"]
            #需要躲避的飞机里面 按照与导弹的距离进行了排序
            dange_evade_list = [value for key, value in sorted(dis_dict.items(), key=lambda d: d[0])]
            for jet in dange_evade_list:
                jam_dis = {}
                for jam in jam_jets:
                    dis = get_dis(jet, jam)
                    if dis not in jam_dis:
                        jam_dis[dis] = jam
                    else:
                        jam_dis[dis+0.1] = jam
                #找到距离最近的有人机
                near_jam_list = [value for key, value in sorted(jam_dis.items(), key=lambda d: d[0])]
                #进行干扰
                for jam in near_jam_list:
                    if (jam not in evade_jets) and (get_dis(jam, jet) < 100000):
                        #转向jet 
                        self.cmd_list.append(CmdEnv.make_areapatrolparam(jam.ID, jam.X, jam.Y, jam.Z, 200, 100, jam.max_speed,
                                            jam.max_acc, jam.max_g))
                        #self.cmd_list.append(CmdEnv.make_followparam(jam.ID, jet.ID, jam.max_speed, jam.max_acc, jam.max_g))
                        self.cmd_list.append(CmdEnv.make_jamparam(jam.ID))
                        #print("我方{}为我方{}提供干扰支持".format(jam.Name,jet.Name))
                        jam_jets.remove(jam)
                        break
        """
    def init_move(self, free_plane, sim_time):
        for my_plane in free_plane:
            if my_plane.Type==2:
                if sim_time%2:
                    relative_pitch = math.pi/5
                else:
                    relative_pitch = -math.pi/5 
            else:
                relative_pitch = 0
            if my_plane.X > self.bound * self.side :
                init_direction = (self.init_direction + 180) % 360
            else:
                if (my_plane.Heading*180/math.pi - self.init_direction)<5 or sim_time<600:
                    init_direction = self.init_direction
                else:
                    init_direction = (self.init_direction + 180) % 360
            distance = 15*my_plane.max_speed
            new_dir = TSVector3.calorientation(init_direction/180*math.pi, relative_pitch)
            new_pos = TSVector3.plus(my_plane.pos3d, TSVector3.multscalar(new_dir, distance))
            if my_plane.Type==1:
                new_pos['Z'] = 6500
                if len(self.my_uav_jets)>0:
                    for enemy in self.enemy_jets:
                        if enemy.Z-60*math.sin(10/180*math.pi)<new_pos['Z']:
                            new_pos['Z'] = enemy.Z-60*math.sin(10/180*math.pi)-100
                if new_pos['Z'] < 2000:
                    new_pos['Z'] = 12500
            route_list = [{"X": new_pos['X'], "Y": new_pos['Y'], "Z": new_pos['Z']},]
            if init_direction != self.init_direction:
                route_list = [{"X": (random.random()*2-1)*3000, "Y": (random.random()*2-1)*3000, "Z": my_plane.Z},]
            # 没有敌人，朝对方中心点移动
            #print('--sim_time---',sim_time)
            if my_plane.Type==1 and sim_time>480:
                self.cmd_list.append(CmdEnv.make_linepatrolparam(my_plane.ID, route_list, 180, 1, 6))
            elif my_plane.Type==2:
                self.cmd_list.append(CmdEnv.make_linepatrolparam(my_plane.ID, route_list, 300, 2, 12))

    def get_threat_target_list(self,demo_flag=False):
        if not demo_flag:
            # 有人机最重要，距离，带弹数量
            threat_dict = {}
            for enemy in self.enemy_jets:
                if enemy.Availability == 0:
                    continue
                dis = 99999999
                for my_plane in self.my_jets:
                    dis_tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                    if dis_tmp < dis:
                        dis = dis_tmp
                enemy.detect_dis = dis
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
            
            for threat_plane in threat_plane_list.copy():
                if threat_plane.Type == 2:
                    #去掉无人机
                    threat_plane_list.remove(threat_plane)
                    continue 
            
                for hit_enemy in self.hit_enemy_list:
                    leader_hit = False
                    # 敌有人机可以打两发
                    if len(hit_enemy[0].locked_missile_list) < 3 and hit_enemy[0].Type == 2:
                        leader_hit = True
                    elif hit_enemy[0].Type == 1:
                        leader_hit = True
                    for threat_plane in threat_plane_list.copy():
                        if hit_enemy[0] == threat_plane and not leader_hit:
                            threat_plane_list.remove(threat_plane)
        else:
            threat_plane_list = []
            for enemy in self.enemy_jets:
                if enemy.IsLost == 0:
                    threat_plane_list.append(enemy)

            if threat_plane_list:
                threat_plane_list = sorted(threat_plane_list, key=lambda d: d.step_valid_missile, reverse=True)
                threat_plane_list = sorted(threat_plane_list, key=lambda d: d.formation_mate, reverse=False)
                threat_plane_list = sorted(threat_plane_list, key=lambda d: d.Type, reverse=False)
        return threat_plane_list
    
    def update_hit_list(self):
        for comba in self.hit_enemy_list:
            if comba[1] is None:
                for my_missile in self.my_missile:
                    if my_missile.EngageTargetID == comba[0].ID:
                        comba[1] = my_missile
                        comba[2] = my_missile.LauncherID
        undetected_list = []
        undetected_id_list = []
        for comba in self.hit_enemy_list.copy():
            my_missile = None
            enemy_plane = comba[0]
            if comba[1] is not None:
                my_missile = comba[1]
            is_dead = False
            if enemy_plane.Availability == 0:
                is_dead = True
            missile_gone = True
            if my_missile is not None and my_missile.IsLost == 0:
                missile_gone = False
            if is_dead or missile_gone:
                self.hit_enemy_list.remove(comba)
            elif not is_dead and not missile_gone:
                if TSVector3.distance(enemy_plane.pos3d, my_missile.pos3d) > 20000*0.8:
                    if enemy_plane.ID not in undetected_id_list:
                        undetected_id_list.append(enemy_plane.ID)
                        undetected_list.append(enemy_plane)
        undetected_list = [comba.ID for comba in sorted(undetected_list, key=lambda d: d.Type, reverse=False)]
        return undetected_list
    
    def can_attack_plane(self, enemy_plane, evade_plane_list, free_plane):
        dis = 9999999
        can_attack_dict = {}
        for my_jet in self.my_jets:
            if my_jet.Availability:
                attack_plane = None
                tmp_dis = TSVector3.distance(my_jet.pos3d, enemy_plane.pos3d)
                if my_jet.Type == 1:
                    left_weapon = my_jet.step_valid_missile > 1
                else:
                    left_weapon = my_jet.step_valid_missile > 0
                in_range = my_jet.can_attack(enemy_plane)
                if (in_range and left_weapon) and tmp_dis < dis:
                    dis = tmp_dis
                    attack_plane = my_jet
                    if attack_plane.ID not in can_attack_dict:
                        can_attack_dict[attack_plane.ID] = tmp_dis
                if attack_plane is not None and attack_plane in evade_plane_list:
                    #若该机还有躲避任务
                    guide_plane = None
                    for free_jet in free_plane:
                        if free_jet.can_detect(enemy_plane):
                            guide_plane = free_jet
                    missile_dis = TSVector3.distance(attack_plane.pos3d, enemy_plane.pos3d)
            
                    enemy_can_detect_me = False
                    for enemy in self.enemy_jets:
                        if enemy.can_detect(attack_plane):
                            enemy_can_detect_me = True
                    if (guide_plane is None or enemy_can_detect_me) and missile_dis > 15000:
                        try:
                            can_attack_dict.pop(attack_plane.ID)
                        except:
                            import pdb;pdb.set_trace()

        can_attack_list = [key for key, value in sorted(can_attack_dict.items(), key=lambda d: d[1])]
        if can_attack_list:
            uav_plane = None
            for plane_id in can_attack_list:
                if self.find_jet_by_given(plane_id, True).Type == 2:
                    uav_plane = plane_id
                    break
            if uav_plane:
                attack_plane = self.find_jet_by_given(uav_plane, True)
            else:
                attack_plane = self.find_jet_by_given(can_attack_list[0], True)
        else:
            attack_plane = None   

        return attack_plane

    def attack(self, sim_time, threat_plane_list, free_plane):
        evade_plane_list = [plane["plane_entity"] for plane in self.evade_list]
        attack_enemy = {}
        for threat_plane in threat_plane_list.copy():
            attack_plane = self.can_attack_plane(threat_plane, evade_plane_list, free_plane)
            for enemy_comba in self.hit_enemy_list:
                if enemy_comba[0] == threat_plane:
                    if threat_plane.ID in attack_enemy:
                        attack_enemy[enemy_comba[0].ID] = max(enemy_comba[3], attack_enemy[enemy_comba[0].ID])
                else:
                    attack_enemy[enemy_comba[0].ID] = enemy_comba[3]
            if attack_plane is not None:
                can_attack_now = True
                if threat_plane.ID in attack_enemy.keys() and sim_time - attack_enemy[threat_plane.ID] < 3:
                    can_attack_now = False
                if can_attack_now:
                    EntityPos = {"X": attack_plane.X, "Y": attack_plane.Y, "Z": attack_plane.Z}
                    EnemyPos = {"X": threat_plane.X, "Y": threat_plane.Y, "Z": threat_plane.Z}
                    distance = TSVector3.distance(EntityPos, EnemyPos)
                    fire_range = (distance + distance/900*threat_plane.max_speed)/attack_plane.launch_range
                    self.cmd_list.append(CmdEnv.make_attackparam(attack_plane.ID, threat_plane.ID, fire_range))
                    attack_plane.step_valid_missile -= 1
                    self.hit_enemy_list.append([threat_plane, None, None, sim_time, None])
                    
                    threat_plane_list.remove(threat_plane)
        return threat_plane_list, free_plane
    
    def attack_guide(self, free_plane, undetected_list):
        for hit_enemy in self.hit_enemy_list:
            enemy_plane = hit_enemy[0]
            guide_plane = None
            if enemy_plane.ID not in undetected_list:
                continue
            tmp_guide_plane = None
            pre_guide_plane = None
            if hit_enemy[4] is not None:
                pre_guide_plane = hit_enemy[4]
            if hit_enemy[2] is not None:
                tmp_guide_plane = self.find_jet_by_given(hit_enemy[2], True)
            if tmp_guide_plane is not None and tmp_guide_plane.can_detect(enemy_plane):
                if tmp_guide_plane in free_plane:
                    hit_enemy[4] = tmp_guide_plane
                    guide_plane = tmp_guide_plane
            elif pre_guide_plane and pre_guide_plane.can_detect(enemy_plane) and pre_guide_plane in free_plane:
                if enemy_plane.ID in undetected_list:
                    guide_plane = pre_guide_plane

            dis = 9999999
            if guide_plane is None:
                for free_jet in free_plane:
                    tmp_dis = TSVector3.distance(free_jet.pos3d, enemy_plane.pos3d)
                    if tmp_dis < dis and free_jet.can_detect(enemy_plane):
                        guide_plane = free_jet
                        dis = tmp_dis
                if guide_plane is not None:
                    hit_enemy[4] = guide_plane

            if guide_plane is not None:
                free_plane.remove(guide_plane)
                undetected_list.remove(enemy_plane.ID)
                if 150<enemy_plane.Speed<300:
                    follow_speed = enemy_plane.Speed
                else:
                    follow_speed = guide_plane.max_speed
                    self.cmd_list.append(CmdEnv.make_followparam(guide_plane.ID, enemy_plane.ID, follow_speed, guide_plane.max_acc, guide_plane.max_g))

    def attack_continue(self, threat_plane_list, free_plane):
        for enemy_plane in threat_plane_list:
            if len(enemy_plane.followed_plane)>0:
                for followed_plane in enemy_plane.followed_plane.copy():
                    if followed_plane in free_plane:
                        if followed_plane.Availability==0:
                            enemy_plane.followed_plane.remove(followed_plane)
                        else:
                            free_plane.remove(followed_plane)
                    else:
                        enemy_plane.followed_plane.remove(followed_plane) 

        if len(free_plane)>0 and len(threat_plane_list)>0:
            for enemy_plane in threat_plane_list:
                if len(enemy_plane.followed_plane)==0 or (len(enemy_plane.followed_plane)<4 and enemy_plane.Type==1):
                    dis = 9999999
                    follow_plane = None
                    for my_plane in free_plane:
                        if my_plane.Type == 1:
                            continue
                        if my_plane.step_valid_missile> 0:
                            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                            if enemy_plane.Type==1:
                                tmp_dis = tmp_dis*0.1
                            elif enemy_plane.LeftWeapon - len(enemy_plane.launched_missile_list) > 0:
                                tmp_dis = tmp_dis*0.5
                            if tmp_dis < dis:
                                follow_plane = my_plane
                                dis = tmp_dis
                    if follow_plane is not None:
                        enemy_plane.followed_plane.append(follow_plane)
                        free_plane.remove(follow_plane)
                for follow_plane in enemy_plane.followed_plane:
                    self.cmd_list.append(CmdEnv.make_followparam(my_plane.ID, enemy_plane.ID, my_plane.max_speed, my_plane.max_acc, my_plane.max_g))

    def update_jet_info(self):
        """更新我方飞机中空闲以及无弹信息"""
        free_plane = []
        zero_missile_plane = []
        for my_jet in self.my_jets:
            if my_jet.Availability and len(my_jet.locked_missile_list):
                if my_jet.Type == 2 and my_jet.step_valid_missile == 0:
                    zero_missile_plane.append(my_jet)
                if my_jet.Type == 1 and my_jet.jam_flag:
                    continue 
                free_plane.append(my_jet)
        free_plane = sorted(free_plane, key=lambda d: d.step_valid_missile, reverse=False)
        free_plane = sorted(free_plane, key=lambda d: d.Type, reverse=True)
        return free_plane, zero_missile_plane

    def assign_mate_planes(self, free_plane, zero_missile_plane):
        if not self.enemy_jets:
            return

        def get_plane_with_min_distance(planes, target_plane, plane_type=None, max_missiles=None):
            min_distance = float("inf")
            selected_plane_id = None
            for plane_id in planes:
                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                distance = TSVector3.distance(plane.pos3d, target_plane.pos3d)

                if plane_type is not None and plane.Type != plane_type:
                    continue
                if max_missiles is not None and plane.step_valid_missile > max_missiles:
                    continue
                if plane.wing_who is not None:
                    continue
                if distance < min_distance:
                    min_distance = distance
                    selected_plane_id = plane_id

            return selected_plane_id

        for leader_plane in self.my_leader_jets:
            if not leader_plane.Availability:
                continue

            mate_plane = leader_plane.formation_mate
            if mate_plane is not None:
                if mate_plane.Availability and leader_plane.locked_missile_list and mate_plane in free_plane:
                    free_plane.remove(mate_plane)
                elif mate_plane not in free_plane or (mate_plane.step_valid_missile and zero_missile_plane):
                    for plane in zero_missile_plane:
                        if plane.can_detect(leader_plane, see_factor=0.9):
                            mate_plane.wing_who = None
                            leader_plane.mate_plane = None
                            break
                    if mate_plane is not None and mate_plane_id in free_plane:
                        free_plane.remove(mate_plane)

            if leader_plane.mate_plane is None:
                selected_plane = get_plane_with_min_distance(zero_missile_plane, leader_plane, plane_type=2)
                if selected_plane is not None:
                    leader_plane.mate_plane = selected_plane
                    zero_missile_plane.remove(selected_plane)
                    free_plane.discard(selected_plane_id)
                else:
                    selected_plane = get_plane_with_min_distance(free_plane, leader_plane, plane_type=2, max_missiles=3)
                    if selected_plane_id is not None:
                        leader_plane.mate_plane = selected_plane_id
                        free_plane.remove(selected_plane_id)

                if leader_plane.mate_plane is not None:
                    self.get_body_info_by_id(self.my_plane, leader_plane.mate_plane).wing_who = leader_plane.ID


    def update_decision(self, sim_time):
        self.cmd_list = []
        self.update_entity_info()
        if sim_time == 3:
            ##三秒以内初始化战斗和位置
            self.init_pos(self.cmd_list)
            #self.init_target()
        elif sim_time > 3:
            evade_plane_list = [plane["plane_entity"] for plane in self.evade_list]
            free_plane, zero_missile_plane = self.update_jet_info()
            self.init_move(free_plane, sim_time)  
            # 为每架飞机指定target
            #self.update_track_target()
            # 更新敌人的被打击列表
            undetected_list = self.update_hit_list()
            # 开火模块
            threat_plane_list = self.get_threat_target_list()
            
            # 更新需要执行躲避的飞机
            self.update_evade()
            #self.protect_leader_plane(sim_time)
            threat_plane_list, free_plane = self.attack(sim_time, threat_plane_list, free_plane)
            
            # 制导
            self.attack_guide(free_plane, undetected_list)

            #躲避
            evade_plane = []
            for comba in self.evade_list:
                my_plane = comba["plane_entity"]
                if my_plane not in evade_plane:
                    evade_plane.append(my_plane)
                else:
                    continue
                enemy_plane = comba["missile_entity"]
                my_plane.evade(enemy_plane, self.cmd_list)
            
            #追击
            #self.attack_continue(threat_plane_list)
            
            
            

                    
            
            #处理一架飞机被多个导弹追踪的情况
            #按照导弹与我机的距离排序
            #每次只处理最近的导弹
            #防止命令覆盖
            '''
            evade_dict = {}
            for comba in self.evade_list:
                plane = comba["plane_entity"]
                missile_entity = comba["missile_entity"]
                if missile_entity is not None:
                    if plane.Name not in evade_dict:
                        evade_dict[plane.Name] = [comba]
                    else:
                        evade_dict[plane.Name].append(comba)
            for evade in evade_dict.keys():
                evade_list = evade_dict[evade]
                dis_dict = {}
                for comb in evade_list:
                    if comb['distance'] not in dis_dict:
                        dis_dict[comb['distance']] = comb["missile_entity"]
                    else:
                        dis_dict[comb['distance']+0.1] = comb["missile_entity"]
                dange_evade_list = [value for key, value in sorted(dis_dict.items(), key=lambda d: d[0])]
                evade_dict[evade] = dange_evade_list

            for evade in evade_dict.keys():
                evade_plane = self.find_jet_by_given(evade)
                enemy = evade_dict[evade][0]
                if evade_plane is not None:
                    evade_plane.evade(sim_time, enemy, self.cmd_list)

                    if evade_plane.Type == 1:
                        for mate in evade_plane.formation_mate:
                            if mate not in evade_plane_list:
                                self.cmd_list.append(
                                    CmdEnv.make_followparam(
                                        mate.ID,
                                        evade_plane.ID,
                                        mate.max_speed,
                                        mate.max_acc,
                                        mate.max_g))
            '''



            '''
            for comba in self.evade_list:
                plane = comba["plane_entity"]
                enemy = comba["missile_entity"]
                #print("导弹：{}，坐标x：{}，坐标y：{},坐标z:{}".format(enemy.Name, enemy.X, enemy.Y, enemy.Z))
                plane.evade(sim_time, enemy, self.cmd_list)
                
                #若是有人机受到导弹追踪，僚机提供雷达支持
                if plane.Type == 1:
                    for mate in plane.formation_mate:
                        if mate not in evade_plane_list:
                            self.cmd_list.append(
                                CmdEnv.make_followparam(
                                    mate.ID,
                                    plane.ID,
                                    mate.max_speed,
                                    mate.max_acc,
                                    mate.max_g))
            '''
                

            
            
            #只有有人机可加干扰  #按危险排序 没有躲避任务且离他最近的有人机执行干扰命令(已去掉)
            ##目前的版本是 有人机把干扰功能留给自己
            #self.update_jam(sim_time)
            


            # 根据target 每架飞机展开行动(飞行 攻击 躲避 制导 干扰)
            #for jet in self.my_jets:
            #    self.tactical_track(jet)
           
            ##部署攻击方案 两种选择1是为每架飞机部署攻击任务 2是为每个有威胁的目标部署攻击任务 
            ##第二种思路更合理一些

            ##更新一个威胁度列表
            #self.get_threat_target_list()


        return self.cmd_list




