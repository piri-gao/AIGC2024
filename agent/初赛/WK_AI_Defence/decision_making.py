import copy
import numpy as np
from env.env_cmd import CmdEnv
import json
from agent.WK_AI_Defence.algorithm.dbscan import DBSCAN
from agent.WK_AI_Defence.utils import get_dis, dir2rad, reg_rad_at, check_and_make_linepatrolparam
from utils.utils_math import TSVector3


class WK_v2_Decision():
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

        ## 编队的攻击角度？？ 110
        self.Fleet_Attack_Angle = 110
        ## 编队的攻击距离？？ 60km
        self.Fleet_Attack_dis = 60000
        ## 协同作战
        self.cooperative_combat_dic = None

        self.first_formation_init_target = {}
        self.sec_formation_init_target = {}

        self.DBSCAN_eps = 25000
        self.DBSCAN_min_sample = 2
        self.start_time = 0


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
            
            self.total_missile_list = step_missile_list
            self.enemy_missile = enemy_missile
            self.my_missile = my_missile

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
        init_direction = 90
        if self.side == -1:
            init_direction = 270
        leader_plane_1 = self.first_leader_formation["leader"]
        leader_plane_2 = self.sec_leader_formation["leader"]
        # 初始化有人机位置
        cmd_list.append(
            CmdEnv.make_entityinitinfo(leader_plane_1.ID, -145000 * self.side, 75000, 9500, 200, init_direction))
        cmd_list.append(
            CmdEnv.make_entityinitinfo(leader_plane_2.ID, -145000 * self.side, -75000, 9500, 200, init_direction))

        for key, value in self.first_leader_formation.items():
            if key == "up_plane":
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(value.ID, -125000 * self.side, 85000, 9500, 200, init_direction))
            if key == "down_plane":
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(value.ID, -125000 * self.side, 65000, 9500, 200, init_direction))

        for key, value in self.sec_leader_formation.items():
            if key == "up_plane":
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(value.ID, -125000 * self.side, -65000, 9500, 200, init_direction))
            if key == "down_plane":
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(value.ID, -125000 * self.side, -85000, 9500, 200, init_direction))

        for i, plane in enumerate(self.first_uav_formation):
            cmd_list.append(
                CmdEnv.make_entityinitinfo(plane.ID, -140000 * self.side, 65000 - ((i + 1) * 10000), 9500, 200,
                                            init_direction))

        for i, plane in enumerate(self.sec_uav_formation):
            cmd_list.append(
                CmdEnv.make_entityinitinfo(plane.ID, -140000 * self.side, -65000 + ((i + 1) * 10000), 9500, 200,
                                            init_direction))

    def init_target(self):
        '''
        前三秒 指定初始目标
        '''
        
        first_formation_route = {"up_plane": {"X": 40000 * self.side, "Y": 85000, "Z": 9500},
                                         "down_plane": {"X": 40000 * self.side, "Y": 65000, "Z": 9500},
                                         "leader_plane": {"X": 25000 * self.side, "Y": 75000, "Z": 9500},
                                         "uav_1": {"X": 30000 * self.side, "Y": 55000, "Z": 9500},
                                         "uav_2": {"X": 30000 * self.side, "Y": 45000, "Z": 9500}
                                         }
        sec_formation_route = {"up_plane": {"X": 40000 * self.side, "Y": -65000, "Z": 9500},
                                "down_plane": {"X": 40000 * self.side, "Y": -85000, "Z": 9500},
                                "leader_plane": {"X": 25000 * self.side, "Y": -75000, "Z": 9500},
                                "uav_1": {"X": 30000 * self.side, "Y": -55000, "Z": 9500},
                                "uav_2": {"X": 30000 * self.side, "Y": -45000, "Z": 9500}
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
                dis = get_dis(enemy_jet, my_jet)
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


    def tactical_track(self, jet):

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
            # 敌机进入攻击距离内
            #print("敌机进入攻击范围")
            pass
            
           

    def update_evade(self):
        ##查看自身是否被导弹锁定
        #遍历敌方导弹 及其目标 让目标做出躲避动作
        #该列表由字典组成，每个字典有两个key:受攻击的我方飞机，攻击我方飞机的敌机导弹
        evade_id = [comb["missile_entity"].ID for comb in self.evade_list]

        # 统计所有被敌方导弹瞄准的我方飞机
        ##目前存在的问题：有可能一架飞机被多个导弹追
        if len(self.enemy_missile) != 0:
            for missile in self.enemy_missile:
                #print("导弹：{}，坐标x：{}，坐标y：{},坐标z:{}".format(missile.Name, missile.X, missile.Y, missile.Z))
                attacked_plane = self.global_observation.observation.get_agent_info_by_id(missile.EngageTargetID)
                if attacked_plane is None:
                    continue
                if missile.ID not in evade_id:
                    evade_comb = {}
                    evade_comb["plane_entity"] = attacked_plane
                    evade_comb["missile_entity"] = missile
                    evade_comb['distance'] = get_dis(attacked_plane, missile)
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
        for evade_plane in self.evade_list:
            attacked_missile = evade_plane["missile_entity"]
            attacked_plane = evade_plane["plane_entity"]
            missile_gone, over_target, safe_distance = False, False, False
            # 导弹已爆炸
            if not self.global_observation.missile_observation.get_missile_info_by_rocket_id(attacked_missile.ID):
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
            
           

        #self.evade_list = evade_list
    
    def update_jam(self, sim_time):
        if sim_time % 61 == 0:
            self.start_time = sim_time
        if sim_time <= (self.start_time + 3):
            jam_jets = [jet for jet in self.my_leader_jets]
            evade_jets = [plane["plane_entity"] for plane in self.evade_list]
            dis_dict = {}
            for comb in self.evade_list:
                if comb['distance'] not in dis_dict:
                    dis_dict[comb['distance']] = comb["plane_entity"]
                else:
                    dis_dict[comb['distance']+0.1] = comb["plane_entity"]
            dange_evade_list = [value for key, value in sorted(dis_dict.items(), key=lambda d: d[0])]
            for jet in dange_evade_list:
                jam_dis = {}
                for jam in jam_jets:
                    dis = get_dis(jet, jam)
                    if dis not in jam_dis:
                        jam_dis[dis] = jam
                    else:
                        jam_dis[dis+0.1] = jam
                near_jam_list = [value for key, value in sorted(jam_dis.items(), key=lambda d: d[0])]

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
    
    def get_threat_target_list(self):
        pass


    def update_decision(self, sim_time):
        self.cmd_list = []
        self.update_entity_info()
        if sim_time == 3:
            ##三秒以内初始化战斗和位置
            self.init_pos(self.cmd_list)
            self.init_target()
        elif sim_time > 3:
            # 为每架飞机指定target
            self.update_track_target()
            
            
            # 更新需要执行躲避的飞机
            self.update_evade()
            for comba in self.evade_list:
                plane = comba["plane_entity"]
                enemy = comba["missile_entity"]
                #print("导弹：{}，坐标x：{}，坐标y：{},坐标z:{}".format(enemy.Name, enemy.X, enemy.Y, enemy.Z))
                plane.evade(sim_time, enemy, self.cmd_list)
            
            
            # 制导
            #只有有人机可加干扰  按危险排序 没有躲避任务且离他最近的有人机执行干扰命令
            ##效果并不明显
            self.update_jam(sim_time)
            


            # 根据target 每架飞机展开行动(飞行 攻击 躲避 制导 干扰)
            for jet in self.my_jets:
                self.tactical_track(jet)
           
            ##部署攻击方案 两种选择1是为每架飞机部署攻击任务 2是为每个有威胁的目标部署攻击任务 
            ##第二种思路更合理一些

            ##更新一个威胁度列表
            self.get_threat_target_list()


        return self.cmd_list




