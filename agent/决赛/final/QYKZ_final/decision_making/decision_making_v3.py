import copy
import numpy as np
from env.env_cmd import CmdEnv
import json
from agent.QYKZ_final.algorithm.dbscan import DBSCAN
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

    def get_free_and_no_missile_planes(my_planes):
        free_planes = []
        no_missile_planes = []

        for plane in my_planes:
            if len(plane.locked_missile_list) == 0 and plane.Availability:
                if plane.Type == 1 and plane.do_jam:
                    continue
                free_planes.append(plane)
                if plane.ready_missile == 0 and plane.Type == 2:
                    no_missile_planes.append(plane.ID)

        free_planes = sorted(free_planes, key=lambda d: (d.Type, d.ready_missile), reverse=(True, False))
        free_planes = [plane.ID for plane in free_planes]

        return free_planes, no_missile_planes

    def get_threat_planes(enemy_planes):
        threat_planes = []

        for enemy in enemy_planes:
            if enemy.lost_flag == 0:
                threat_planes.append(enemy.ID)

        threat_planes = sorted(threat_planes, key=lambda d: (
            self.get_body_info_by_id(enemy_planes, d).Type,
            self.get_body_info_by_id(enemy_planes, d).wing_plane,
            -self.get_body_info_by_id(enemy_planes, d).ready_missile
        ))

        return threat_planes

    def assign_wing_planes(free_planes, no_missile_planes, leader_planes):
        for leader in leader_planes:
            if leader.Availability:
                wing_plane_id = leader.wing_plane

                if wing_plane_id is not None:
                    wing_plane = self.get_body_info_by_id(self.my_plane, wing_plane_id)

                    if wing_plane.Availability and len(leader.locked_missile_list) > 0:
                        if wing_plane.ID in free_planes:
                            free_planes.remove(wing_plane.ID)
                            continue
                        else:
                            for plane_id in free_planes:
                                plane = self.get_body_info_by_id(self.my_plane, plane_id)
                                if plane.can_see(leader, see_factor=0.9):
                                    wing_plane.wing_who = None
                                    leader.wing_plane = None
                                    break
                            if wing_plane.wing_who is not None:
                                continue
                    if wing_plane.ID not in free_planes:
                        wing_plane.wing_who = None
                        leader.wing_plane = None
                    elif wing_plane.ready_missile != 0 and len(no_missile_planes) > 0:
                        for plane_id in no_missile_planes:
                            plane = self.get_body_info_by_id(self.my_plane, plane_id)
                            if plane.can_see(leader, see_factor=0.9):
                                wing_plane.wing_who = None
                                leader.wing_plane = None
                                break
                    if leader.wing_plane is not None and wing_plane.ID in free_planes:
                        free_planes.remove(wing_plane.ID)

                if leader.wing_plane is None:
                    min_distance = float('inf')
                    best_plane_id = None

                    for plane_id in no_missile_planes + free_planes:
                        plane = self.get_body_info_by_id(self.my_plane, plane_id)
                        distance = TSVector3.distance(plane.pos3d, leader.pos3d)

                        if distance < min_distance and plane.Type == 2 and plane.wing_who is None:
                            min_distance = distance
                            best_plane_id = plane_id

                    if best_plane_id is not None:
                        leader.wing_plane = best_plane_id
                        self.get_body_info_by_id(self.my_plane, best_plane_id).wing_who = leader.ID
                        no_missile_planes.discard(best_plane_id)
                        free_planes.discard(best_plane_id)
                        ##--------------------------------------------------------------------

    def get_jam_for_attack(my_leader_plane, sim_time):
        jam_for_attack = []
        for leader in my_leader_plane:
            if leader.Availability and sim_time - leader.last_jam > 60 and len(leader.locked_missile_list) == 0:
                jam_for_attack.append(leader)
        return jam_for_attack

    def find_jam_targets(jam_plane, my_missile, enemy_planes):
        can_jammed_enemy = {}
        jammed_missile_list = []
        for missile_id in my_missile.copy():
            missile = self.get_body_info_by_id(self.missile_list, missile_id)
            target_plane = self.get_body_info_by_id(enemy_planes, missile.EngageTargetID)
            jam_turn_time = self.get_turn_time(jam_plane, target_plane)
            can_jam = jam_turn_time * missile.Speed < TSVector3.distance(missile.pos3d, target_plane.pos3d) < (jam_turn_time + 6) * missile.Speed
            can_see = TSVector3.distance(jam_plane.pos3d, target_plane.pos3d) < 1200000 - jam_turn_time * target_plane.Speed
            if target_plane.ID not in can_jammed_enemy and missile.lost_flag == 0 and target_plane.lost_flag == 0 and can_jam and missile.loss_target == False and can_see:
                can_jammed_enemy[target_plane.ID] = jam_plane.pi_bound(TSVector3.calheading(TSVector3.minus(target_plane.pos3d, jam_plane.pos3d)) - jam_plane.Heading)
                my_missile.remove(missile_id)
                jammed_missile_list.append(missile_id)
        return sorted(can_jammed_enemy.keys(), key=lambda x: can_jammed_enemy[x]), jammed_missile_list

    def execute_jamming(jam_plane, middle_enemy, sim_time, cmd_list):
        if jam_plane.can_see(middle_enemy, see_factor=0.6, jam_dis=117600):
            jam_plane.do_jam = True
            cmd_list.append(env_cmd.make_jamparam(jam_plane.ID))
            jam_plane.jammed = sim_time + 1
            jam_plane.middle_enemy_plane = None
            jam_plane.last_jam = sim_time + 6
        else:
            jam_enemy = self.get_body_info_by_id(self.enemy_plane, jam_plane.middle_enemy_plane)
            turn_time = self.get_turn_time(jam_plane, jam_enemy)
            can_jam_after_turn = False
            for missile_id in jammed_missile_list:
                missile = self.get_body_info_by_id(self.missile_list, missile_id)
                if missile.arrive_time >= turn_time + 3:
                    can_jam_after_turn = True
            if can_jam_after_turn:
                self.find_way_jam(jam_plane, jam_plane.middle_enemy_plane, cmd_list)
    

    def calculate_missile_num(self):
        if self.my_score > self.enemy_score:
            if self.my_score > self.enemy_score + 36:
                return 4
            else:
                return 1
        else:
            return 0

    def update_attack_enemy(threat_plane, attack_enemy):
        for missile_id in threat_plane.locked_missile_list:
            missile = self.get_body_info_by_id(self.missile_list, missile_id)
            if threat_ID in attack_enemy:
                attack_enemy[threat_ID] = max(missile.fire_time, attack_enemy[threat_ID])
            else:
                attack_enemy[threat_ID] = missile.fire_time

    def check_attack_conditions(self, attack_plane, threat_plane, attack_enemy):
        can_attack_now = True
        attack_dis = TSVector3.distance(attack_plane.pos3d, threat_plane.pos3d)
        cold_time = calculate_cold_time(self, threat_plane, attack_dis)

        if not is_within_cold_time(self, threat_plane, attack_enemy, cold_time):
            can_attack_now = False

        if is_score_based_condition_met(self):
            can_attack_now = False

        return can_attack_now, cold_time

    def calculate_cold_time(self, threat_plane, attack_dis):
        if self.my_score < self.enemy_score + 36:
            cold_time = 0
            if threat_plane.Type == 2:
                return cold_time
        else:
            if threat_plane.Type == 2:
                if attack_dis < 6000:
                    cold_time = 1
                else:
                    cold_time = 4
            else:
                cold_time = 3
        return cold_time

    def is_within_cold_time(self, threat_plane, attack_enemy, cold_time):
        threat_ID = threat_plane.ID
        return threat_ID in attack_enemy.keys() and self.sim_time - attack_enemy[threat_ID] < cold_time

    def is_score_based_condition_met(self):
        return (self.my_score - 1 == self.enemy_score and self.my_center_time < self.enemy_center_time) or \
            (self.my_score == self.enemy_score and (self.my_center_time > self.enemy_center_time * 2 or \
                self.my_center_time > self.enemy_center_time + (20 * 60 - self.sim_time) * self.live_enemy_leader))

    def update_decision(self, sim_time):
        self.cmd_list = []
        self.update_entity_info()
        if sim_time == 3:
            ##三秒以内初始化战斗和位置
            self.init_pos(self.cmd_list)
            #self.init_target()
        elif sim_time > 3:
            free_planes, no_missile_planes = self.get_free_and_no_missile_planes(self.my_plane)
            threat_planes = self.get_threat_planes(self.enemy_plane)
            self.assign_wing_planes(free_planes, no_missile_planes, self.my_leader_plane)

            ###干扰模块

            jam_for_attack = get_jam_for_attack(self.my_leader_plane, self.sim_time)
            if jam_for_attack:
                my_missile = self.get_body_info_by_identification(self.missile_list, self.my_plane[0].Identification)
                for jam_plane in jam_for_attack:
                    can_jammed_enemy, jammed_missile_list = find_jam_targets(jam_plane, my_missile, self.enemy_plane)
                    if can_jammed_enemy:
                        if jam_plane.middle_enemy_plane is None:
                            jam_plane.middle_enemy_plane = can_jammed_enemy[int(len(can_jammed_enemy) / 2)]
                        middle_enemy = self.get_body_info_by_id(self.enemy_plane, jam_plane.middle_enemy_plane)
                        execute_jamming(jam_plane, middle_enemy, self.sim_time, cmd_list)
            
            #攻击模块
            attack_enemy = {}
            for threat_plane_id in threat_plane_list:
                threat_plane = self.get_body_info_by_id(self.enemy_plane, threat_plane_id)
                missile_num = calculate_missile_num()

                if threat_plane.Type == 2 and len(threat_plane.locked_missile_list) > missile_num:
                    continue

                threat_ID = threat_plane.ID
                attack_plane = self.can_attack_plane(threat_plane)
                update_attack_enemy(threat_plane, attack_enemy)

                if attack_plane is not None:
                    can_attack_now, cold_time = check_attack_conditions(self, attack_plane, threat_plane, attack_enemy)
                    if can_attack_now:
                        factor_fight = 1
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_ID, factor_fight))
                        attack_plane.ready_missile -= 1
                        self.my_score -= 1
            
            #yo



        return self.cmd_list




