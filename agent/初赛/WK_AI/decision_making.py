import copy
import numpy as np

from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3

class WK_Decision():

    def __init__(self, global_observation):

        self.global_observation = global_observation

        self.init_info()

    def init_info(self):
        # 我方所有飞机信息列表
        self.my_plane = []
        # 我方有人机信息列表
        self.my_leader_plane = []
        # 我方无人机信息列表
        self.my_uav_plane =[]

        # 敌方所有飞机信息列表
        self.enemy_plane = []
        # 敌方有人机信息列表
        self.enemy_leader_plane = []
        # 敌方无人机信息列表
        self.enemy_uav_plane = []

        # 我方导弹信息
        self.my_missile = []
        # 敌方导弹信息
        self.enemy_missile = []

        # 被打击的敌机列表
        self.hit_enemy_list = []
        self.hit_enemy_dict = {}

        # 识别红军还是蓝军
        self.side = None


        # 有人机第一编队
        self.first_leader_formation = {}
        # 有人机第二编队
        self.sec_leader_formation = {}

        # 无人机第一编队
        self.first_uav_formation = [0, 0]
        # 无人机第二编队
        self.sec_uav_formation = [0, 0]

        # 第一编队
        self.first_formation = {}
        # 第二编队
        self.sec_formation = {}

        # 躲避列表
        self.evade_list = []
        self.choice_mode = False
        self.placeholder = False

    def reset(self):
        """当引擎重置会调用,选手需要重写此方法"""
        self.init_info()

    def update_entity_info(self, sim_time):
        # 己方所有飞机信息
        self.my_plane = self.global_observation.observation.get_all_agent_list()
        # 己方有人机信息
        self.my_leader_plane = self.global_observation.observation.get_agent_info_by_type(1)
        # 己方无人机信息
        self.my_uav_plane = self.global_observation.observation.get_agent_info_by_type(2)
        # 敌方所有飞机信息
        self.enemy_plane = self.global_observation.perception_observation.get_all_agent_list()
        # 敌方有人机信息
        self.enemy_leader_plane = self.global_observation.perception_observation.get_agent_info_by_type(1)
        # 敌方无人机信息
        self.enemy_uav_plane = self.global_observation.perception_observation.get_agent_info_by_type(2)

        # 获取队伍标识
        if self.side is None:
            if self.my_plane[0].Identification == "红方":
                self.side = 1
            else:
                self.side = -1

        # 获取双方导弹信息
        missile = self.global_observation.missile_observation.get_missile_list()
        enemy_missile = []
        my_missile = []
        for rocket in missile:
            # 待测试
            if rocket.Identification == self.my_plane[0].Identification:
                my_missile.append(rocket)
            else:
                enemy_missile.append(rocket)
        self.enemy_missile = enemy_missile
        self.my_missile = my_missile
        #print("发现{}架敌机".format(len(self.enemy_plane)))
        #print("发现{}发敌方导弹".format(len(self.enemy_missile)))
        # 编队并更新编队信息（待测试）
        # self.formation()

    def formation(self):
        self.first_leader_formation["up_plane"] = self.my_uav_plane[0]
        self.first_leader_formation["down_plane"] = self.my_uav_plane[1]
        self.first_leader_formation["leader"] = self.my_leader_plane[0]
        self.sec_leader_formation["up_plane"] = self.my_uav_plane[2]
        self.sec_leader_formation["down_plane"] = self.my_uav_plane[3]
        self.sec_leader_formation["leader"] = self.my_leader_plane[1]

        self.first_uav_formation[0] = self.my_uav_plane[4]
        self.first_uav_formation[1] = self.my_uav_plane[5]

        self.sec_uav_formation[0] = self.my_uav_plane[6]
        self.sec_uav_formation[1] = self.my_uav_plane[7]

        self.first_formation["up_plane"] = self.my_uav_plane[0]
        self.first_formation["down_plane"] = self.my_uav_plane[1]
        self.first_formation["leader_plane"] = self.my_leader_plane[0]
        self.first_formation["uav_1"] = self.my_uav_plane[4]
        self.first_formation["uav_2"] = self.my_uav_plane[5]

        self.sec_formation["up_plane"] = self.my_uav_plane[2]
        self.sec_formation["down_plane"] = self.my_uav_plane[3]
        self.sec_formation["leader_plane"] = self.my_leader_plane[1]
        self.sec_formation["uav_1"] = self.my_uav_plane[6]
        self.sec_formation["uav_2"] = self.my_uav_plane[7]

        # 干扰
        self.jam_list = [[plane, 0] for plane in self.my_leader_plane]

    def update_decision(self, sim_time, cmd_list):
        #print("当前仿真时间:",sim_time)
        self.update_entity_info(sim_time)
        if sim_time <= 2:
            self.init_pos(sim_time, cmd_list)
        else:
            self.init_move(cmd_list)

            # 更新打击列表
            self.update_hit_dict()


            # 开火模块
            threat_plane_list = self.get_threat_target_list()
            if sim_time > 450:
                for threat_plane in threat_plane_list:
                    #if threat_plane.Type == 1:
                        #print("发现敌方有人机:",threat_plane.Name)
                    attack_plane = self.can_attack_plane(threat_plane)
                    if attack_plane is not None:
                        #临近有制导机在20公里以内才发射
                        if threat_plane.guide_plane is not None:
                            dis = TSVector3.distance(threat_plane.pos3d, threat_plane.guide_plane.pos3d)
                            if dis <= 20000:
                                if threat_plane.ID not in self.hit_enemy_dict:
                                    if attack_plane.Type == 1:
                                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 0.8))
                                    else:
                                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 1))
                                    #print("我方飞机:{}打击敌方飞机:{}".format(attack_plane.Name, threat_plane.Name))
                                    attack_plane.num_left_missile -= 1
                                    self.hit_enemy_dict[threat_plane.ID] = [None]
                                    threat_plane.num_locked_missile += 1
                                    threat_plane_list.remove(threat_plane)
                                #if (threat_plane.ID in self.hit_enemy_dict) and (threat_plane.Type == 1) and (len(self.hit_enemy_dict[threat_plane.ID])==1):
                                # 敌机为有人机 且仅被一发导弹锁定
                                elif threat_plane.ID in self.hit_enemy_dict:
                                    if (threat_plane.Type == 2) and (len(self.hit_enemy_dict[threat_plane.ID])<2):
                                        if attack_plane.Type == 1:
                                            cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 0.8))
                                        else:
                                            cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 1))
                                        #print("我方飞机:{}打击敌方飞机:{}".format(attack_plane.Name, threat_plane.Name))
                                        attack_plane.num_left_missile -= 1
                                        self.hit_enemy_dict[threat_plane.ID].append(None)
                                        threat_plane.num_locked_missile += 1
                                        threat_plane_list.remove(threat_plane)
                #print("打击列表:",self.hit_enemy_dict)


            # 制导
            #evade_list是字典
            #evade_comb["plane_entity"] 存储我方被导弹锁定的飞机
            # evade_comb["missile_entity"] = missile 存储锁定我方飞机的导弹
            #evade_list.append(evade_comb)
            #获取所有需要进行躲避的我方飞机

            evade_plane_id = [plane["plane_entity"].ID for plane in self.evade_list]
            if (450<sim_time<470) and (self.first_leader_formation["leader"].ID in evade_plane_id):
                self.placeholder = True
            if sim_time == 480 and (not self.placeholder):
                self.choice_mode = True
                #print("----------------------------------------------------------------------------")
            #undetected_list中存储的是被我方锁定过 但是躲避成功的飞机

            for enemy_plane in threat_plane_list:
                free_plane = []
                for my_plane in self.my_plane:
                    if my_plane.ID not in evade_plane_id and my_plane.Availability:
                        #若我方飞机没被敌机导弹锁定 则可执行制导
                        free_plane.append(my_plane)
                dis = 999999
                guide_plane = None

                for my_plane in free_plane:
                    tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
                    if tmp_dis < dis:
                        guide_plane = my_plane
                        dis = tmp_dis

                if guide_plane is not None:
                    z_point = enemy_plane.Z
                    if z_point > 10000:
                        z_point = 10000
                    if z_point < 2000:
                        z_point = 2000
                    #print("我方执行制导的飞机:{},用于标记制导的敌机:{}".format(guide_plane.Name,enemy_plane.Name))
                    guide_plane.guide_stats = True
                    cmd_list.append(env_cmd.make_areapatrolparam(guide_plane.ID, enemy_plane.X, enemy_plane.Y,
                                                                 z_point, 200, 100, 300, 1, 6))
                    #cmd_list.append(env_cmd.make_followparam(guide_plane.ID, enemy_plane.ID, 300, 1, 6))
                    #指定制导飞机
                    enemy_plane.guide_plane = guide_plane

            my_plane_name = [plane.Name for plane in self.my_plane]
            eva_plane_name = [plane["plane_entity"].Name for plane in self.evade_list]
            threat_plane_name = [plane.Name for plane in threat_plane_list]
            if len(eva_plane_name) >0 or len(threat_plane_name) > 0:
                pass
                #print("我方此刻还存活飞机:{},执行躲避任务的飞机:{},我方待打击的飞机:{}".format(my_plane_name,eva_plane_name,threat_plane_name))

            # 躲避
            self.update_evade()
            for comb in self.evade_list:
                attacked_missile = comb["missile_entity"]
                attacked_plane = comb["plane_entity"]
                if attacked_plane in self.my_plane:
                    attacked_plane.evade(attacked_missile, cmd_list)

            #干扰
            self.activate_jam(cmd_list)



    def init_pos(self, sim_time, cmd_list):
        self.formation()
        # 初始化部署
        if sim_time == 2:
            init_direction = 90
            if self.side == -1:
                init_direction = 270
            leader_plane_1 = self.first_leader_formation["leader"]
            leader_plane_2 = self.sec_leader_formation["leader"]
            # 初始化有人机位置
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -145000 * self.side, 75000, 9500, 200, init_direction))
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_2.ID, -145000 * self.side, -75000, 9500, 200, init_direction))

            for key, value in self.first_leader_formation.items():
                if key == "up_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, 85000, 9500, 200, init_direction))
                if key == "down_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, 65000, 9500, 200, init_direction))

            for key, value in self.sec_leader_formation.items():
                if key == "up_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -65000, 9500, 200, init_direction))
                if key == "down_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -85000, 9500, 200, init_direction))

            for i, plane in enumerate(self.first_uav_formation):
                cmd_list.append(
                    env_cmd.make_entityinitinfo(plane.ID, -140000 * self.side, 65000 - ((i + 1) * 10000), 9500, 200,
                                                init_direction))

            for i, plane in enumerate(self.sec_uav_formation):
                cmd_list.append(
                    env_cmd.make_entityinitinfo(plane.ID, -140000 * self.side, -65000 + ((i + 1) * 10000), 9500, 200,
                                                init_direction))


    def init_move(self, cmd_list):
        # 无目标移动
        enemy_leader_list = []
        evade_plane_list = [plane["plane_entity"] for plane in self.evade_list]
        for enemy_plane in self.enemy_plane:
            if enemy_plane.Type == 1:
                enemy_leader_list.append(enemy_plane)
        if len(enemy_leader_list) == 1:
            #print("发现敌方一架有人机:",enemy_leader_list[0].Name)
            if enemy_leader_list[0].Y >= 0:
                for p, plane in self.sec_formation.items():
                    #if plane not in evade_plane_list:
                    if self.choice_mode:
                        cmd_list.append(env_cmd.make_followparam(plane.ID, enemy_leader_list[0].ID, 300, 1, 6))

                    # if plane.Type == 1:
                    #     cmd_list.append(env_cmd.make_followparam(plane.ID, enemy_leader_list[0].ID, 300, 1, 6))
                    # if plane.Type == 2:
                    #     cmd_list.append(env_cmd.make_followparam(plane.ID, self.sec_leader_formation["leader"].ID, 300, 1, 6))
            else:
                for p, plane in self.first_formation.items():
                    #if plane not in evade_plane_list:
                    if self.choice_mode:
                        cmd_list.append(env_cmd.make_followparam(plane.ID, enemy_leader_list[0].ID, 300, 1, 6))

                    # if plane.Type == 1:
                    #     cmd_list.append(env_cmd.make_followparam(plane.ID, enemy_leader_list[0].ID, 300, 1, 6))
                    # if plane.Type == 2:
                    #     cmd_list.append(
                    #         env_cmd.make_followparam(plane.ID, self.first_leader_formation["leader"].ID, 300, 1, 6))
        elif len(enemy_leader_list) == 2:
            #若发现两架有人机
            #print(("发现敌方2架有人机:",enemy_leader_list[0].Name))
            dis = 9999999
            for enemy_leader_plane in enemy_leader_list:
                for my_leader_plane in self.my_leader_plane:
                    tmp_dis = TSVector3.distance(my_leader_plane.pos3d, enemy_leader_plane.pos3d) 
                    if tmp_dis < dis:
                        leader_hit = my_leader_plane
                        dis = tmp_dis

        if len(self.enemy_plane) <= 1:
            if len(self.enemy_plane) == 1:
                if self.enemy_plane[0].Y >= 0:
                    for p, plane in self.sec_formation.items():
                        cmd_list.append(env_cmd.make_followparam(plane.ID, self.first_formation[p].ID, 300, 1, 6))
                else:
                    for p, plane in self.first_formation.items():
                        cmd_list.append(env_cmd.make_followparam(plane.ID, self.sec_formation[p].ID, 300, 1, 6))
            else:
                # 没有敌人，朝中心点移动
                first_formation_route = {"up_plane": [{"X": 40000 * self.side, "Y": 85000, "Z": 9500}],
                                         "down_plane": [{"X": 40000 * self.side, "Y": 65000, "Z": 9500}],
                                         "leader_plane": [{"X": 25000 * self.side, "Y": 75000, "Z": 9500}],
                                         "uav_1": [{"X": 30000 * self.side, "Y": 55000, "Z": 9500}],
                                         "uav_2": [{"X": 30000 * self.side, "Y": 45000, "Z": 9500}]
                                         }
                sec_formation_route = {"up_plane": [{"X": 40000 * self.side, "Y": -65000, "Z": 9500}],
                                       "down_plane": [{"X": 40000 * self.side, "Y": -85000, "Z": 9500}],
                                       "leader_plane": [{"X": 25000 * self.side, "Y": -75000, "Z": 9500}],
                                       "uav_1": [{"X": 30000 * self.side, "Y": -55000, "Z": 9500}],
                                       "uav_2": [{"X": 30000 * self.side, "Y": -45000, "Z": 9500}]
                                       }
                for plane, route in first_formation_route.items():
                    cmd_list.append(env_cmd.make_linepatrolparam(self.first_formation[plane].ID, route, 240, 1, 6))
                for plane, route in sec_formation_route.items():
                    cmd_list.append(env_cmd.make_linepatrolparam(self.sec_formation[plane].ID, route, 240, 1, 6))



    def _is_dead(self, plane):
        if plane.Availability == 0:
            return True
        else:
            return False

    def update_evade(self):
        ##查看自身是否被导弹锁定
        #遍历敌方导弹 及其目标 让目标做出躲避动作
        evade_list = copy.deepcopy(self.evade_list)
        evade_id = [comb["missile_entity"].ID for comb in evade_list]

        # 统计所有被敌方导弹瞄准的我方飞机
        if len(self.enemy_missile) != 0:
            for missile in self.enemy_missile:
                attacked_plane = self.global_observation.observation.get_agent_info_by_id(missile.EngageTargetID)
                if attacked_plane is None:
                    continue
                if missile.ID not in evade_id:
                    evade_comb = {}
                    evade_comb["plane_entity"] = attacked_plane
                    evade_comb["missile_entity"] = missile
                    evade_list.append(evade_comb)
            # 给危险程度分类 TODO

        #判断危险是否解除
        #若危险解除则再次返航
        for evade_plane in evade_list:
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
            missile_mp_vector_3d = TSVector3.minus(attacked_plane.pos3d, attacked_missile.pos3d)
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
            #distance = TSVector3.distance(attacked_missile.pos3d, attacked_plane.pos3d)
            ##满足任何一种情况：已爆炸、过靶、飞到安全距离 都移除 即不需要再躲避 如何区分是我方的呢？？
            if any([missile_gone, over_target, safe_distance]):
            #if any([missile_gone, over_target]) and safe_distance:
                evade_list.remove(evade_plane)

        self.evade_list = evade_list



    def update_attack(self):
        pass

    def get_threat_target_list(self):
        # 有人机最重要，距离，带弹数量
        threat_dict = {}
        for enemy in self.enemy_plane:
            dis = 99999999
            for my_plane in self.my_plane:
                dis_tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                if dis_tmp < dis:
                    dis = dis_tmp
                if enemy.Type == 1:
                    dis -= 20000
            '''            
            for my_plane in self.my_leader_plane:
                dis_tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                if dis_tmp < dis:
                    dis = dis_tmp
            if enemy.Type == 1:
                # 敌机在距离我方有人机在距离的前提下会多20000的威胁值，并且敌人是有人机会再多10000威胁值
                dis -= 10000
            dis -= 20000

            for my_plane in self.my_uav_plane:
                dis_tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                if dis_tmp < dis:
                    dis = dis_tmp
            '''
            if dis < 0:
                dis = 0
            if dis not in threat_dict:
                threat_dict[dis] = enemy
            else:
                threat_dict[dis + 0.1] = enemy

        threat_plane_list = [value for key, value in sorted(threat_dict.items(), key=lambda d: d[0])]
        threat_list = [key for key, value in sorted(threat_dict.items(), key=lambda d: d[0])]
        threat_plane_name_list = [value.Name for key, value in sorted(threat_dict.items(), key=lambda d: d[0])]
        #print("敌机的威胁度排名:",threat_plane_name_list)
        #print("敌机的威胁度为:",threat_list)

        for key, value in self.hit_enemy_dict.items():
            for threat_plane in threat_plane_list:
                if threat_plane.ID == key:
                    #无人机最多打两发
                    if (threat_plane.Type == 2) and (threat_plane.num_locked_missile >= 1):
                        threat_plane_list.remove(threat_plane)
        '''
        for hit_enemy in self.hit_enemy_list:
            leader_hit = False
            # 敌有人机可以打两发
            if hit_enemy[0].num_locked_missile == 1 and hit_enemy[0].Type == 1:
                leader_hit = True
            for threat_plane in threat_plane_list:
                if hit_enemy[0] == threat_plane and not leader_hit:
                    threat_plane_list.remove(threat_plane)
        '''
        return threat_plane_list

    def can_attack_plane(self, enemy_plane):
        attack_plane = None
        dis = 9999999
        for my_plane in self.my_plane:
            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
            # if my_plane.Type == 1:
            #     left_weapon = my_plane.LeftWeapon > 1
            # else:
            #     left_weapon = my_plane.LeftWeapon > 0
            left_weapon = my_plane.num_left_missile > 0
            in_range = my_plane.can_attack(tmp_dis)
            if (in_range and left_weapon) and tmp_dis < dis:
                dis = tmp_dis
                attack_plane = my_plane

        return attack_plane

    def update_hit_dict(self):
        #test

        # print_hit = {}
        # for key,value in self.hit_enemy_dict.items():
        #     #enemy_temp = self.global_observation.perception_observation.get_agent_info_by_id(key)
        #     #这里无法使用查询函数 因为有些敌机可能已经躲避攻击了
        #     for enemy in self.enemy_plane:
        #         if enemy.ID == key:
        #             print_hit[enemy.Name] = value
        # print("敌方飞机受攻击情况:",print_hit)


        for enemy_id,attacked_missile_list in self.hit_enemy_dict.items():
            missile_list = []
            if None in attacked_missile_list:
                for my_missile in self.my_missile:
                    if my_missile.EngageTargetID == enemy_id:
                        missile_list.append(my_missile)
                self.hit_enemy_dict[enemy_id] = missile_list

        undetected_list = []
        for enemy_id in list(self.hit_enemy_dict.keys()):
            attacked_missile_list = self.hit_enemy_dict[enemy_id]

            is_dead = False
            if self.global_observation.perception_observation.get_agent_info_by_id(enemy_id):
                #还能查询到敌机 就判断是否死亡
                enemy_temp = self.global_observation.perception_observation.get_agent_info_by_id(enemy_id)
                is_dead = self._is_dead(enemy_temp)
                if is_dead:
                    #如果已经死了 移除该敌机 直接下一个
                    #print("敌方飞机:{}已被击落".format(enemy_temp.Name))
                    #对应飞机的制导任务也解除
                    enemy_temp.guide_plane.guide_stats = False
                    self.hit_enemy_dict.pop(enemy_id)
                    continue
                else:
                    if None not in attacked_missile_list:
                        for missile in attacked_missile_list:
                            if not self.global_observation.missile_observation.get_missile_info_by_rocket_id(
                                    missile.ID):
                                # 该导弹已爆炸 直接移除
                                attacked_missile_list.remove(missile)
                                self.hit_enemy_dict[enemy_id] = attacked_missile_list

            else:
                #直接查询不到敌机 应该是是躲避打击了
                #有个疑惑 为啥查不到敌机了 还可以获取到3d位置
                #enemy_temp = self.global_observation.perception_observation.get_agent_info_by_id(enemy_id)
                #这里也不能用查询函数
                # for enemy in self.enemy_plane: 这里有问题 既然查询不到那么self.enemy_plane 也查不到
                #     if enemy.ID == enemy_id:
                #由于之前dict存储的是id 这里已经读取不到3d了 并且使用undetected_list提供制导目标也不是很合理
                #直接改为使用受攻击目标
                #undetected_list.append(enemy,enemy.pos3d)
                #查询不到信息的也打击列表里移除 意味着下次重新侦测到可以继续打击
                self.hit_enemy_dict.pop(enemy_id)

        #return undetected_list

    def activate_jam(self, cmd_list):
        for comba in self.jam_list:
            if comba[1] > 10:
                comba[1] = 0
                cmd_list.append(env_cmd.make_jamparam(comba[0].ID))
            else:
                comba[1] += 1









