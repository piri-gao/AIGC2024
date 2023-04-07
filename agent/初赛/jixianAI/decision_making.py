""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/8/22 16:43
"""


import copy
import numpy as np
from env.env_cmd import CmdEnv
from agent.jixianAI.utils import get_dis, dir2rad, reg_rad_at, check_and_make_linepatrolparam
from agent.jixianAI.algorithm.dbscan import DBSCAN
from agent.jixianAI.agent_base import EnemyJet
import json

class JixianXsimAI():

    def __init__(self, global_observation):
        self.global_observation = global_observation
        with open("agent/jixianAI/agent.json") as f:
            self.config = json.load(f)
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

        self.Fleet_Attack_Angle = self.config["formation_attack"]["attack_angle"]

        self.Fleet_Attack_dis = self.config["formation_attack"]["attack_dis"]

        # 协同作战
        self.cooperative_combat_dic = None
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
                my_missile.append(missile)
            else:
                enemy_missile.append(missile)

            missile_active = (missile.ID != 0 and missile.Availability > 0.0001)
            if not missile_active:
                continue
            launch_id = missile.LauncherID
            target_id = missile.EngageTargetID
            launch_jet = self.find_jet_by_given(launch_id, True)
            target_jet = self.find_jet_by_given(target_id, True)
            if launch_jet is not None:
                launch_jet.my_launch_missile.append(missile)
            if target_jet is not None:
                target_jet.attacked_missile_list.append(missile)

            # 如果这发导弹不是新导弹，那么它之前的step就已经被添加进self.misssile
            # 如果find不到返回none，那么证明这是个新发射的导弹
            new_missile = self.find_missile_by_id(missile.ID)
            if new_missile is None:
                # 如果发射这枚导弹的飞机有EnemyLeftWeapon属性，证明是敌机，用这一步来计算敌方导弹剩余
                if launch_jet is not None and hasattr(launch_jet, "EnemyLeftWeapon"):
                    launch_jet.EnemyLeftWeapon -= 1
                if launch_jet is not None:
                    # 什么时候从这个列表里清楚这个正在飞行的导弹还没写
                    launch_jet.my_launch_missile.append(new_missile)

        self.total_missile_list = step_missile_list
        self.enemy_missile = enemy_missile
        self.my_missile = my_missile

        # 更新编队信息（待开发）todo


    def _init_battle(self):
        # # 初始化编队
        # self.first_leader_formation["up_jet"] = self.my_jets_dic["红无人机1"]
        # self.first_leader_formation["down_jet"] = self.my_jets_dic["红无人机2"]
        # self.first_leader_formation["leader"] = self.my_jets_dic["红有人机1"]
        # self.sec_leader_formation["up_jet"] = self.my_jets_dic["红有人机3"]
        # self.sec_leader_formation["down_jet"] = self.my_jets_dic["红无人机4"]
        # self.sec_leader_formation["leader"] = self.my_jets_dic["红有人机2"]
        #
        # self.first_uav_formation[0] = self.my_jets_dic["红无人机5"]
        # self.first_uav_formation[1] = self.my_jets_dic["红无人机6"]
        #
        # self.sec_uav_formation[0] = self.my_jets_dic["红无人机7"]
        # self.sec_uav_formation[1] = self.my_jets_dic["红无人机8"]
        #
        # self.first_formation["up_jet"] = self.first_leader_formation["up_jet"]
        # self.first_formation["down_jet"] = self.first_leader_formation["down_jet"]
        # self.first_formation["leader_jet"] = self.first_leader_formation["leader"]
        # self.first_formation["uav_1"] = self.first_uav_formation[0]
        # self.first_formation["uav_2"] = self.first_uav_formation[1]
        #
        # self.sec_formation["up_jet"] = self.sec_leader_formation["up_jet"]
        # self.sec_formation["down_jet"] = self.sec_leader_formation["down_jet"]
        # self.sec_formation["leader_jet"] = self.sec_leader_formation["leader"]
        # self.sec_formation["uav_1"] = self.sec_uav_formation[0]
        # self.sec_formation["uav_2"] = self.sec_uav_formation[1]

        if self.side == 1:
            color = "红"
        else:
            color = "蓝"
        formation_1st = {
            color + "有人机1":{
                "squad": "L1",
                "formation_mate": [color + "无人机1", color + "无人机2"]
            },
            color + "无人机1": {
                "squad": "L1",
                "formation_mate": [color + "无人机2", color + "有人机1"]
            },
            color + "无人机2": {
                "squad": "L1",
                "formation_mate": [color + "无人机1", color + "有人机1"]
            },
            color + "无人机3": {
                "squad": "U1",
                "formation_mate": [color + "无人机4"]
            },
            color + "无人机4": {
                "squad": "U1",
                "formation_mate": [color + "无人机3"]
            },
        }

        formation_2nd = {
            color + "有人机2":{
                "squad": "L2",
                "formation_mate": [color + "无人机5", color + "无人机6"]
            },
            color + "无人机5": {
                "squad": "L2",
                "formation_mate": [color + "无人机6", color + "有人机2"]
            },
            color + "无人机6": {
                "squad": "L2",
                "formation_mate": [color + "无人机5", color + "有人机2"]
            },
            color + "无人机7": {
                "squad": "U2",
                "formation_mate": [color + "无人机8"]
            },
            color + "无人机8": {
                "squad": "U2",
                "formation_mate": [color + "无人机7"]
            },
        }

        for jet in self.my_jets:
            if jet.formation == "first":
                jet.squad = formation_1st[jet.Name]["squad"]
                jet.formation_mate = formation_1st[jet.Name]["formation_mate"]
            else:
                jet.squad = formation_2nd[jet.Name]["squad"]
                jet.formation_mate = formation_2nd[jet.Name]["formation_mate"]
        self.formation_1st = formation_1st
        self.formation_2nd = formation_2nd

        # 构建假目标，供战斗初级我方战机朝敌方飞行
        fake_target1 = copy.deepcopy(self.my_jets[0])
        fake_target2 = copy.deepcopy(self.my_jets[0])
        fake_target1.X, fake_target2.X = self.config["init_target_up"]["X"], self.config["init_target_down"]["X"]
        fake_target1.Y, fake_target2.Y = self.config["init_target_up"]["Y"], self.config["init_target_down"]["Y"]
        fake_target1.Z, fake_target2.Z = self.config["init_target_up"]["Z"], self.config["init_target_down"]["Z"]
        fake_target1.pos2d, fake_target2.pos2d = np.array([fake_target1.X, fake_target1.Y], dtype=float), np.array(
            [fake_target2.X, fake_target2.Y], dtype=float)
        fake_target1.pos3d, fake_target2.pos3d = np.array([fake_target1.X, fake_target1.Y, fake_target1.Z],
                                                          dtype=float), np.array(
            [fake_target2.X, fake_target2.Y, fake_target2.Z], dtype=float)
        fake_target1.Identification, fake_target1.Identification = "假", "假"

        self.fake_target1 = fake_target1
        self.fake_target2 = fake_target2

    def update_decision(self, sim_time):
        self.cmd_list = []
        self.update_entity_info()
        if sim_time == 3:
            self._init_battle()
            self.init_pos()
        elif sim_time > 3:

            # 第一次更新我方飞机当前的目标
            self.update_persistent_state()
            self.update_track_target()





            for jet in self.my_jets:
                if jet.current_status == "tactical_track":
                    # tactical_track需要一个目标，所以要在执行这个动作前，更新要锁定的目标
                    self.tactical_track(jet)

        return self.cmd_list



    def init_pos(self):
        init_direction = 90
        if self.side == -1:
            init_direction = 270

        if self.side == 1:
            pre = "红"
        elif self.side == -1:
            pre = "蓝"

        # 初始化有人机位置
        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"有人机1"].ID, -145000 * self.side, 75000, 9500, 200, init_direction))
        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"有人机2"].ID, -145000 * self.side, -75000, 9500, 200, init_direction))

        # 初始化无人机位置
        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"无人机1"].ID, -125000 * self.side, 85000, 9500, 200, init_direction))

        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"无人机2"].ID, -125000 * self.side, 65000, 9500, 200,
                                        init_direction))

        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"无人机3"].ID, -125000 * self.side, 55000, 9500, 200,
                                        init_direction))

        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"无人机4"].ID, -125000 * self.side, 45000, 9500, 200,
                                        init_direction))

        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"无人机5"].ID, -125000 * self.side, -85000, 9500, 200,
                                        init_direction))

        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"无人机6"].ID, -125000 * self.side, -65000, 9500, 200,
                                        init_direction))

        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"无人机7"].ID, -125000 * self.side, -55000, 9500, 200,
                                        init_direction))

        self.cmd_list.append(
            CmdEnv.make_entityinitinfo(self.my_jets_dic[pre+"无人机8"].ID, -125000 * self.side, -45000, 9500, 200,
                                        init_direction))

    def update_track_target(self):# todo
        """
        选择目标：
            初始目标：假目标


        无目标移动:
            有人机编队移动
            无人机编队移动
        有目标追踪
        """
        # 依据敌人之间互相的位置对敌人进行聚类
        enemy_pos_cluster = self.enemy_pos_cluster()


        if len(self.enemy_jets) == 0:
            print("未探测到敌人，执行编队朝假目标飞行任务")
            for my_jet in self.my_jets:
                my_jet.current_status = "tactical_track"
                if my_jet.formation == "first":
                    my_jet.target = self.fake_target1
                else:
                    my_jet.target = self.fake_target2
        else:
            print("已探测到敌人")
            for cluster in self.enemy_pos_list:
                # 找到离当前cluster最近我方编队, 和当前cluster距离最近的敌人，这里可能需要考虑对这个cluster做一个威胁排序
                attack_jet, target = self.get_nearest_jet(cluster, self.my_jets)
                if attack_jet is not None:
                    squad = attack_jet.squad
                    for jet in self.my_jets:
                        if jet.squad == squad:
                            jet.target = target
                            if jet.Type == 1:
                                self.cmd_list.append(CmdEnv.make_attackparam(jet.ID, target.ID, 0.8))
                            else:
                                self.cmd_list.append(CmdEnv.make_attackparam(jet.ID, target.ID, 1))



    def tactical_track(self, jet):
        # 无目标时，朝假目标列队飞行
        jet_mates = [self.find_jet_by_given(name) for name in jet.formation_mate]
        # 过滤掉已经阵亡的队友
        jet_mates = [mate for mate in jet_mates if mate is not None]
        # 更新当前jet的编队队友列表
        jet.formation_mate = [mate.Name for mate in jet_mates]

        # 选择track目标
        # 如果没有敌人则把初始目标点当作敌人, 进行未接敌的组队移动
        init_move = jet.target.Identification == "假"
        attack_dis = get_dis(jet, jet.target)

        if init_move or attack_dis > self.Fleet_Attack_dis:
            self.formation_track(jet)

        elif attack_dis < self.Fleet_Attack_dis:

            follow = hasattr(jet, 'target') and (jet.target is not None)
            if not follow:
                return

            nolaunched_missile = len([ms for ms in self.my_missile if ms.LauncherID == jet.ID]) == 0


            # 列举一些特殊情况，这下情况下我方战机不会正常跟随敌机
            # 我方这架战机基本没用了，那么朝着敌方有人机飞
            if jet.LeftWeapon == 0 and nolaunched_missile and jet.Type == 2 and jet.target.Type == 1 and get_dis(jet, jet.target)<14e3:
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
                        speed= 400
                    elif threat_distance < 40e3:
                        speed = 350
                    else:
                        speed = 300

                    dest_location = [{
                        "X": jet.target.X, "Y": jet.target.Y, "Z": jet.max_alt*0.9
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


        """
        如果敌人数量小于5，那么只派最近的队伍去靠近
        另一只队伍仍然朝着假目标移动
          
        不分敌人数量
        
        首先每个敌人都是单独的个体，在选择敌人的时候，假如用最简单的逻辑，只选择距离最近的敌机
        那么对敌机进行划分，每个类只派我方一个队伍去攻击：每个类敌人数量为3
        然后剩下的队伍对异常值计算距离，按顺序排序去攻击
        
        为什么每个类只派一队去攻击，
        
        1队
        2队
        3队
        4队
        工作到：dbscan划分有问题！！！！！！！！
        
        
        假如打到后面敌人数量又小于5了，但是目标不会再是假目标了

        """

    def formation_track(self, jet):
        """
        列队呈包夹队形朝敌人飞行

        Args:
            jet: jet object

        Returns:

        """
        member_num = len(jet.formation_mate) + 1

        # 向量target->my_plane
        delta = jet.pos2d - jet.target.pos2d

        uav_mates = [self.find_jet_by_given(mate) for mate in jet.formation_mate]

        # 向量集合 target->mate_plane
        delta_all = [mate.pos2d - jet.target.pos2d for mate in uav_mates]
        delta_all.append(delta)

        # 获得当前战机的小队中心位置坐标
        squad_center = jet.pos2d / member_num
        for mate in uav_mates:
            squad_center += mate.pos2d / member_num

        # 计算小队成员的弧度差排序
        rad_basement = dir2rad(squad_center - jet.target.pos2d)
        rad_all = [reg_rad_at(dir2rad(d) - rad_basement, 0)
                   for d in delta_all]
        rank_index = np.argsort(rad_all)
        rank_at = np.where(rank_index == (member_num - 1))[0][0]

        # 根据当前飞机在小队中的位置，计算目标点
        if member_num == 1:
            # 如果小队成员只有自己，那么就朝着目标敌人飞去
            dest_now3d = jet.target.pos3d
        else:
            attack_rad = self.Fleet_Attack_Angle * (np.pi / 180)
            jet_rad = rad_basement - attack_rad / 2 + rank_at * (attack_rad / (member_num - 1))
            vec2d = self.Fleet_Attack_dis * np.array([np.cos(jet_rad), np.sin(jet_rad)])
            dest_now3d = copy.deepcopy(jet.target.pos3d)
            dest_now3d[:2] += vec2d

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
            dis_error, min_x=-3e3, y_min_x=jet.max_speed*0.5,
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

    def update_persistent_state(self):
        for my_jet in self.my_jets:
            if my_jet.check_ms_number():
                my_jet.persistent_status = "tactical_track"



    def find_jet_by_given(self, name_or_id, id=False):
        for jet in self.my_jets + self.enemy_jets:
            if id == True:
                if jet.ID == name_or_id:
                    return jet
            else:
                if jet.Name == name_or_id:
                    return jet
        return None

    def find_missile_by_id(self, id):
        for missile in self.total_missile_list:
            if missile.ID == id:
                return missile
        return None

    def battle_situation(self):
        enemy_num = len(self.enemy_jets)
        enemy_area = [enemy_jet.Y > 0 for enemy_jet in self.enemy_jets]

        return 1

    def enemy_pos_cluster(self):
        """
        如果敌人数量多于3个，那么对他们进行聚类分析，如果只有1个类

        Returns:

        """

        enemy_pos = [[jet.X, jet.Y] for jet in self.enemy_jets]
        enemy_pos_cluster = DBSCAN(enemy_pos, self.config["DBSCAN"]["eps"], self.config["DBSCAN"]["min_sample"])

        labels = []
        enemy_pos_list = []

        for pos_c, jet in enumerate(self.enemy_jets):
            label = enemy_pos_cluster[pos_c]
            if label not in labels and label != -1:
                labels.append(label)
                enemy_pos_list.append([])
                enemy_pos_list[label - 1].append(jet)
            elif label in labels:
                enemy_pos_list[label-1].append(jet)

        for pos_c, jet in enumerate(self.enemy_jets):
            if enemy_pos_cluster[pos_c]  == -1:
                enemy_pos_list.append([jet])

        self.enemy_pos_list = enemy_pos_list

        return enemy_pos_list

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

    def get_nearest_jet(self, jets_list1, jets_list2):
        """
        从两个战机集合中，依据距离找到最近的一对最近的战机

        Args:
            jets_list1:
            jets_list2:

        Returns:
            两个战机实体

        """
        min_dis = 99999999
        attack_jet = None
        attacked_jet  = None
        for enemy_jet in jets_list1:
            for my_jet in jets_list2:
                dis = get_dis(enemy_jet, my_jet)
                if dis < min_dis:
                    min_dis = dis
                    attack_jet = my_jet
                    attacked_jet = enemy_jet
        if attack_jet and attacked_jet:
            return attack_jet, attacked_jet
        else:
            return None, None



