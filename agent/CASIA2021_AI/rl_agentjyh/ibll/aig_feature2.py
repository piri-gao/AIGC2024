# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-10-04 07:56:15
# @Last Modified by:   Your name
# @Last Modified time: 2021-10-12 13:12:28
import math
import numpy as np

class Feature:
    def __init__(self):
        """对下文中使用到的变量进行定义和初始化"""
        self.my_uvas_infos = []  # 该变量用以保存己方所有无人机的态势信息
        self.my_manned_info = []  # 该变量用以保存己方有人机的态势信息
        self.my_allplane_infos = []  # 该变量用以保存己方所有飞机的态势信息
        self.my_missile_infos = []  # 该变量用以保存我方导弹的态势信息
        self.my_allplane_maps = {}  # 我方所有飞机映射字典  ID -> 飞机信息
        self.my_allmissile_maps = {}  # 我方所有导弹映射字典  ID -> 导弹信息

        self.enemy_uvas_infos = []  # 该变量用以保存敌方所有无人机的态势信息
        self.enemy_manned_info = []  # 该变量用以保存敌方有人机的态势信息
        self.enemy_allplane_infos = []  # 该变量用以保存敌方所有飞机的态势信息
        self.enemy_missile_infos = []  # 该变量用以保存敌方导弹的态势信息
        self.enemy_allplane_maps = {}  # 敌方所有飞机映射字典  ID -> 飞机信息
        self.enemy_allmissile_maps = {}  # 敌方所有导弹映射字典  ID -> 导弹信息

        self.enemy_leftweapon = {}  # 该变量用以保存敌方飞机的导弹量

        self.missile_path_length = {}  # 导弹的已飞路程
        self.missile_last_pisition = {}  # 导弹上个时刻的位置
        self.missile_launch_time = {}  # 导弹的发射时间

        self.all_plane_map = {}  # 所有飞机的映射字典  虚拟ID -> 飞机信息
        self.all_id_map = {}  # 所有飞机id的映射字典  虚拟ID -> 真实ID
        self.all_id_map_reverse = {}  # 所有飞机id的逆映射字典  真实ID -> 虚拟ID
        self.plane_key_info = {}  # 所有飞机的关键信息  虚拟ID -> 飞机的关键信息

        self.all_missile_map = {}  # 所有导弹的映射字典  虚拟ID -> 导弹信息
        self.all_missile_id_map = {}  # 所有导弹id的映射字典  虚拟ID -> 真实ID
        self.all_missile_id_map_reverse = {}  # 所有导弹id的逆映射字典  真实ID -> 虚拟ID

        self.planeID_2_missileID = {}  # 所有飞机的对应的导弹虚拟ID  飞机虚拟ID -> list(导弹虚拟ID)
        self.plane_has_launched = {}  # 所有飞机的已经发射的导弹的信息  飞机真实ID -> list(导弹真实ID)
        self.EPS = 1e-8
        
    def _dis(self, a, b):
        tmp = {"X": a["X"] - b["X"], "Y": a["Y"] - b["Y"], "Z": a["Z"] - b["Z"]}
        if tmp["X"] == 0 and tmp["Y"] == 0 and tmp["Z"] == 0:
            return 0
        return math.sqrt(tmp["X"] * tmp["X"] + tmp["Y"] * tmp["Y"] + tmp["Z"] * tmp["Z"])

    def process_observation(self, obs_side, sim_time):
        """
        处理飞机态势信息
        :param obs_side: 当前方所有的态势信息信息，包含了所有的当前方所需信息以及探测到的敌方信息
        """
        my_entity_infos = obs_side['platforminfos'] # 拿到己方阵营有人机、无人机在内的所有飞机信息
        my_manned_info = [] # 用以保存当前己方有人机信息
        my_uvas_infos = []  # 用以保存当前己方无人机信息
        my_allplane_infos = []    # 用以保存当前己方所有飞机信息
        for uvas_info in my_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001: # 判断飞机是否可用 飞机的ID即为飞机的唯一编号 飞机的Availability为飞机当前生命值
                uvas_info["Z"] = uvas_info["Alt"]     # 飞机的 Alt 即为飞机的当前高度
                if uvas_info['Type'] == 1:           # 所有类型为 1 的飞机是 有人机
                    my_manned_info.append(uvas_info) # 将有人机保存下来 一般情况，每方有人机只有1架
                if uvas_info['Type'] == 2:           # 所有类型为 2 的飞机是 无人机
                    my_uvas_infos.append(uvas_info)  # 将无人机保存下来 一般情况，每方无人机只有4架
                my_allplane_infos.append(uvas_info)        # 将己方所有飞机信息保存下来 一般情况，每方飞机实体总共5架

        enemy_entity_infos = obs_side['trackinfos']   # 拿到敌方阵营的飞机信息,包括敌方有人机、无人机在内的所有飞机信息
        enemy_manned_info = []  # 用以保存当前敌方有人机信息
        enemy_uvas_infos = []   # 用以保存当前敌方无人机信息
        enemy_allplane_infos = []     # 用以保存当前敌方所有飞机信息
        for uvas_info in enemy_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001:  # 判断飞机是否可用 飞机的ID即为飞机的唯一编号 飞机的Availability为飞机当前生命值
                uvas_info['Z'] = uvas_info['Alt']         # 飞机的 Alt 即为飞机的当前高度
                if uvas_info['Type'] == 1:               # 所有类型为 1 的飞机是 有人机
                    enemy_manned_info.append(uvas_info)  # 将有人机保存下来 一般情况，每方有人机只有1架
                if uvas_info['Type'] == 2:               # 所有类型为 2 的飞机是 无人机
                    enemy_uvas_infos.append(uvas_info)   # 将无人机保存下来 一般情况，每方无人机只有4架
                enemy_allplane_infos.append(uvas_info)         # 将己方所有飞机信息保存下来 一般情况，每方飞机实体总共5架

        self.my_allplane_maps = {}
        for input_entity in my_allplane_infos:
            self.my_allplane_maps[int(input_entity['ID'])] = input_entity

        self.enemy_allplane_maps = {}
        for input_entity in enemy_allplane_infos:
            self.enemy_allplane_maps[int(input_entity['ID'])] = input_entity

        missile_infos = obs_side['missileinfos']  # 拿到空间中已发射且尚未爆炸的导弹信息

        my_missile_infos = []       # 用以保存己方已发射且尚未爆炸的导弹信息
        enemy_missile_infos = []    # 用以保存敌方已发射且尚未爆炸的导弹信息
        
        if sim_time == 2:
            self.my_plane_maps = self.my_allplane_maps

        for missile_info in missile_infos:
            if (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001): # 判断导弹是否可用 导弹的ID即为导弹的唯一编号 导弹的Availability为导弹当前生命值
                if missile_info['LauncherID'] in self.my_plane_maps:  # 判断导弹是否为己方导弹 导弹的LauncherID即为导弹的发射者
                    missile_info["Z"] = missile_info["Alt"]     # 导弹的 Alt 即为导弹的当前高度
                    my_missile_infos.append(missile_info)       # 保存己方已发射且尚未爆炸的导弹信息
                else:
                    missile_info["Z"] = missile_info["Alt"]     # 导弹的 Alt 即为导弹的当前高度
                    enemy_missile_infos.append(missile_info)    # 保存敌方已发射且尚未爆炸的导弹信息

        self.my_allmissile_maps = {}
        for input_entity in my_missile_infos:
            self.my_allmissile_maps[int(input_entity['ID'])] = input_entity

        self.enemy_allmissile_maps = {}
        for input_entity in enemy_missile_infos:
            self.enemy_allmissile_maps[int(input_entity['ID'])] = input_entity

        for enemy_plane in enemy_allplane_infos:  # 初始化敌方飞机当前载弹量
            if enemy_plane['ID'] not in self.enemy_leftweapon:
                if enemy_plane['Type'] == 1:
                    self.enemy_leftweapon[enemy_plane['ID']] = 4
                else:
                    self.enemy_leftweapon[enemy_plane['ID']] = 2

        old_missile_ID_list = []
        for old_missile in self.my_missile_infos + self.enemy_missile_infos:
            old_missile_ID_list.append(old_missile['ID'])

        for missile in my_missile_infos + enemy_missile_infos:  # 对导弹信息进行统计
            if missile['ID'] not in old_missile_ID_list:  # 出现新导弹
                self.missile_path_length[missile['ID']] = 0  # 初始化已走导弹路径长度
                self.missile_last_pisition[missile['ID']] = {'X': missile['X'], 'Y': missile['Y'], 'Z': missile['Z']}  # 初始化上时刻位置
                self.missile_launch_time[missile['ID']] = missile['CurTime']  # 统计导弹发射时间
                if missile['LauncherID'] in self.enemy_leftweapon:  # 更新敌方飞机的当前载弹量
                    self.enemy_leftweapon[missile['LauncherID']] -= 1
            else:
                self.missile_path_length[missile['ID']] += self._dis(
                    self.missile_last_pisition[missile['ID']], missile)  # 更新已走导弹路径长度
            self.missile_last_pisition[missile['ID']]['X'] = missile['X']  # 更新上时刻位置
            self.missile_last_pisition[missile['ID']]['Y'] = missile['Y']
            self.missile_last_pisition[missile['ID']]['Z'] = missile['Z']

        self.my_uvas_infos = my_uvas_infos              # 保存当前己方无人机信息
        self.my_manned_info = my_manned_info            # 保存当前己方有人机信息
        self.my_allplane_infos = my_allplane_infos            # 保存当前己方所有飞机信息
        self.my_missile_infos = my_missile_infos     # 保存己方已发射且尚未爆炸的导弹信息
        self.enemy_uvas_infos = enemy_uvas_infos        # 保存当前敌方无人机信息
        self.enemy_manned_info = enemy_manned_info      # 保存当前敌方有人机信息
        self.enemy_allplane_infos = enemy_allplane_infos      # 保存当前敌方所有飞机信息
        self.enemy_missile_infos = enemy_missile_infos  # 保存敌方已发射且尚未爆炸的导弹信息

    def process_planemap(self, sim_time):
        if sim_time == 2:
            # 初始化所有飞机的字典映射信息
            my_plane_map = {plane["ID"]: plane for plane in self.my_allplane_infos}
            my_order = sorted([id for id in my_plane_map.keys()])
            for index, id in enumerate(my_order):
                self.all_plane_map[index] = my_plane_map[id]
                self.all_id_map[index] = my_plane_map[id]["ID"]
                self.all_id_map_reverse[my_plane_map[id]["ID"]] = index
                self.plane_key_info[index] = my_plane_map[id]

            enemy_plane_map = {plane["ID"]: plane for plane in self.enemy_allplane_infos}
            enemy_order = sorted([id for id in enemy_plane_map.keys()])
            for index, id in enumerate(enemy_order):
                self.all_plane_map[index + 5] = enemy_plane_map[id]
                self.all_id_map[index + 5] = enemy_plane_map[id]["ID"]
                self.all_id_map_reverse[enemy_plane_map[id]["ID"]] = index + 5
                self.plane_key_info[index + 5] = enemy_plane_map[id]

            for i in range(10):
                temp_dict = {}
                temp_dict["Identification"] = self.plane_key_info[i]["Identification"]
                temp_dict["ID"] = self.plane_key_info[i]["ID"]
                temp_dict["Type"] = self.plane_key_info[i]["Type"]
                temp_dict["Availability"] = self.plane_key_info[i]["Availability"]
                self.plane_key_info[i] = temp_dict

            temp = 0
            for i in range(10):
                self.planeID_2_missileID[i] = []
                if self.plane_key_info[i]["Type"] == 1:
                    num_missiles = 4
                else:
                    num_missiles = 2
                for _ in range(num_missiles):
                    self.planeID_2_missileID[i].append(temp)
                    temp += 1

        elif sim_time > 2:
            # 初始化所有飞机的字典映射信息
            my_plane_map = {plane["ID"]: plane for plane in self.my_allplane_infos}
            for planeID, myplane in my_plane_map.items():
                false_planeID = self.all_id_map_reverse[planeID]
                self.all_plane_map[false_planeID] = myplane

            enemy_plane_map = {plane["ID"]: plane for plane in self.enemy_allplane_infos}
            for planeID, enemyplane in enemy_plane_map.items():
                false_planeID = self.all_id_map_reverse[planeID]
                self.all_plane_map[false_planeID] = enemyplane
            
            # 删除阵亡飞机的字典映射信息
            alive_ids = [plane["ID"] for plane in self.my_allplane_infos + self.enemy_allplane_infos]
            for index, plane in self.all_plane_map.items():
                if plane is not None and plane["ID"] not in alive_ids:
                    self.all_plane_map[index] = None
                    self.plane_key_info[index]["Availability"] = 0

    def process_missilemap(self, sim_time):
        for i in range(24):
            self.all_missile_map[i] = None
        if sim_time == 2:
            for plane in self.my_allplane_infos + self.enemy_allplane_infos:
                self.plane_has_launched[plane['ID']] = []
        for missile in self.my_missile_infos + self.enemy_missile_infos:
            plane_id = missile['LauncherID']
            plane_false_id = self.all_id_map_reverse[plane_id]
            if missile['ID'] not in self.plane_has_launched[plane_id]:
                self.plane_has_launched[plane_id].append(missile['ID'])
            i_pos = self.plane_has_launched[plane_id].index(missile['ID'])
            pos = plane_false_id * 2 + i_pos
            if plane_false_id > 0:
                pos += 2
            if plane_false_id > 5:
                pos += 2
            self.all_missile_map[pos] = missile

    def get_current_control_info(self, cur_index):
        cur_plane = self.plane_key_info[cur_index]
        Identification_my = np.array([0]) if cur_plane["Identification"] == "红方" else np.array([1])
        Type_my = np.array([0]) if cur_plane["Type"] == 1 else np.array([1])
        Availability_my = np.array([cur_plane["Availability"]])
        X_my = np.array([0])
        Y_my = np.array([0])
        Z_my = np.array([0])
        Heading_my = np.array([0])
        Pitch_my = np.array([0])
        Roll_my = np.array([0])
        Speed_my = np.array([0])
        AccMag_my = np.array([0])
        NormalG_my = np.array([0])
        IsLocked_my = np.array([0])
        LeftWeapon_my = np.array([0])
        cur_plane = self.all_plane_map[cur_index]
        if cur_plane is not None:
            X_my = -np.array([cur_plane["X"]])
            Y_my = np.array([cur_plane["Y"]])
            Z_my = np.array([cur_plane["Z"]])
            Heading_my = -np.array([cur_plane["Heading"]])
            Pitch_my = np.array([cur_plane["Pitch"]])
            Roll_my = -np.array([cur_plane["Roll"]])
            Speed_my = np.array([cur_plane["Speed"]])
            AccMag_my = np.array([cur_plane["AccMag"]])
            NormalG_my = np.array([cur_plane["NormalG"]])
            IsLocked_my = np.array([cur_plane["IsLocked"]])
            LeftWeapon_my = np.array([cur_plane["LeftWeapon"]])
        current_control_info = np.concatenate((Identification_my, Type_my, Availability_my, X_my,
                                               Y_my, Z_my, Heading_my, Pitch_my, Roll_my, Speed_my,
                                               AccMag_my, NormalG_my, IsLocked_my, LeftWeapon_my))

        # 归一化处理
        cur_plane = self.plane_key_info[cur_index]
        current_control_info[3] /= 150000  # X_my:[-1, 1]
        current_control_info[4] /= 150000  # Y_my:[-1, 1]
        current_control_info[5] = (current_control_info[5] - 2000) / 13000 if cur_plane["Type"] == 1 else (current_control_info[5] - 2000) / 8000  # Z_my:[0, 1]
        current_control_info[6] /= math.pi  # Heading_my:[-1, 1]
        current_control_info[7] /= (math.pi/2)  # Pitch_my:[-1, 1]
        current_control_info[8] /= math.pi  # Roll_my:[-1, 1]
        current_control_info[9] = (current_control_info[9] - 150) / 250 if cur_plane["Type"] == 1 else (current_control_info[9] - 100) / 200  # Speed:[0, 1]
        current_control_info[10] /= (1 if cur_plane["Type"] == 1 else 2) * 9.80665  # AccMag:[0, 1]
        current_control_info[11] /= (6 if cur_plane["Type"] == 1 else 12) * 9.80665  # NormalG:[0, 1]
        current_control_info[13] /= (4 if cur_plane["Type"] == 1 else 2)  # LeftWeapon_my:[0, 1]

        return current_control_info

    def get_my_allplane_info(self, cur_index, my_allplane_info):
        if my_allplane_info is not None:
            cur_plane = self.all_plane_map[cur_index]
            for i in range(5):
                plane = self.all_plane_map[i]
                Relative_X_i = np.array([0])
                Relative_Y_i = np.array([0])
                Relative_Z_i = np.array([0])
                Relative_dist_i = np.array([0])
                Relative_theta_i = np.array([0])
                Relative_alpha_i = np.array([0])
                if plane is not None and cur_plane is not None and i != cur_index:
                    Relative_X_i = -np.array([plane["X"] - cur_plane["X"]])
                    Relative_Y_i = np.array([plane["Y"] - cur_plane["Y"]])
                    Relative_Z_i = np.array([plane["Z"] - cur_plane["Z"]])
                    Relative_dist_i = np.array([self._dis(plane, cur_plane)])
                    Relative_theta_i = np.array([self.pi_bound(self.XY2theta(Relative_X_i[0], Relative_Y_i[0]))])
                    Relative_alpha_i = np.array([self.pi_bound(math.asin(Relative_Z_i[0] / (Relative_dist_i[0] + self.EPS)))])

                    # 归一化处理
                    Relative_X_i[0] /= 300000  # Relative_X_i:[-1, 1]
                    Relative_Y_i[0] /= 300000  # Relative_Y_i:[-1, 1]
                    Relative_Z_i[0] /= 13000  # Relative_Z_i:[-1, 1]
                    Relative_dist_i[0] /= 424436  # Relative_dist_i:[0, 1]
                    Relative_theta_i[0] /= math.pi  # Relative_theta_i:[-1, 1]
                    Relative_alpha_i[0] /= (math.pi / 2)  # Relative_alpha_i:[-1, 1]

                my_allplane_info[i*22+14] = Relative_X_i
                my_allplane_info[i*22+15] = Relative_Y_i
                my_allplane_info[i*22+16] = Relative_Z_i
                my_allplane_info[i*22+17] = Relative_dist_i
                my_allplane_info[i*22+18] = Relative_theta_i
                my_allplane_info[i*22+19] = Relative_alpha_i
            return my_allplane_info

        myplane_info = []
        for i in range(5):
            plane = self.plane_key_info[i]
            Identification_i = np.array([0]) if plane["Identification"] == "红方" else np.array([1])
            Type_i = np.array([0]) if plane["Type"] == 1 else np.array([1])
            Availability_i = np.array([plane["Availability"]])
            X_i = np.array([0])
            Y_i = np.array([0])
            Z_i = np.array([0])
            Heading_i = np.array([0])
            Pitch_i = np.array([0])
            Roll_i = np.array([0])
            Speed_i = np.array([0])
            AccMag_i = np.array([0])
            NormalG_i = np.array([0])
            IsLocked_i = np.array([0])
            LeftWeapon_i = np.array([0])
            Relative_X_i = np.array([0])
            Relative_Y_i = np.array([0])
            Relative_Z_i = np.array([0])
            Relative_dist_i = np.array([0])
            Relative_theta_i = np.array([0])
            Relative_alpha_i = np.array([0])
            Is_man_and_incenter_i = np.array([0])
            Center_dist_i = np.array([212247])
            plane = self.all_plane_map[i]
            cur_plane = self.all_plane_map[cur_index]
            if plane is not None:
                X_i = -np.array([plane["X"]])
                Y_i = np.array([plane["Y"]])
                Z_i = np.array([plane["Z"]])
                Heading_i = -np.array([plane["Heading"]])
                Pitch_i = np.array([plane["Pitch"]])
                Roll_i = -np.array([plane["Roll"]])
                Speed_i = np.array([plane["Speed"]])
                AccMag_i = np.array([plane["AccMag"]])
                NormalG_i = np.array([plane["NormalG"]])
                IsLocked_i = np.array([plane["IsLocked"]])
                LeftWeapon_i = np.array([plane["LeftWeapon"]])
                if i != cur_index and cur_plane is not None:
                    Relative_X_i = -np.array([plane["X"] - cur_plane["X"]])
                    Relative_Y_i = np.array([plane["Y"] - cur_plane["Y"]])
                    Relative_Z_i = np.array([plane["Z"] - cur_plane["Z"]])
                    Relative_dist_i = np.array([self._dis(plane, cur_plane)])
                    Relative_theta_i = np.array([self.pi_bound(self.XY2theta(Relative_X_i[0], Relative_Y_i[0]))])
                    Relative_alpha_i = np.array([self.pi_bound(math.asin(Relative_Z_i[0] / (Relative_dist_i[0] + self.EPS)))])
                if plane["Type"] == 1 and self.is_in_center(plane):
                    Is_man_and_incenter_i = np.array([1])
                if plane["Type"] == 1:
                    if self.is_in_center(plane):
                        Center_dist_i = np.array([0])
                    else:
                        distance_to_center = (plane['X'] ** 2 + plane["Y"] ** 2 + (plane["Alt"] - 9000) ** 2) ** 0.5
                        Center_dist_i = np.array([distance_to_center - 50000])
            whole_info_i = np.concatenate((Identification_i, Type_i, Availability_i, X_i,
                                           Y_i, Z_i, Heading_i, Pitch_i, Roll_i, Speed_i,
                                           AccMag_i, NormalG_i, IsLocked_i, LeftWeapon_i,
                                           Relative_X_i, Relative_Y_i, Relative_Z_i, Relative_dist_i,
                                           Relative_theta_i, Relative_alpha_i, Is_man_and_incenter_i,
                                           Center_dist_i))
            # 归一化处理
            plane = self.plane_key_info[i]
            whole_info_i[3] /= 150000  # X_i:[-1, 1]
            whole_info_i[4] /= 150000  # Y_i:[-1, 1]
            whole_info_i[5] = (whole_info_i[5] - 2000) / 13000 if plane["Type"] == 1 else (whole_info_i[5] - 2000) / 8000  # Z_i:[0, 1]
            whole_info_i[6] /= math.pi  # Heading_i:[-1, 1]
            whole_info_i[7] /= (math.pi / 2)  # Pitch_i:[-1, 1]
            whole_info_i[8] /= math.pi  # Roll_i:[-1, 1]
            whole_info_i[9] = (whole_info_i[9] - 150) / 250 if plane["Type"] == 1 else (whole_info_i[9] - 100) / 200  # Speed_i:[0, 1]
            whole_info_i[10] /= (1 if plane["Type"] == 1 else 2) * 9.80665  # AccMag_i:[0, 1]
            whole_info_i[11] /= (6 if plane["Type"] == 1 else 12) * 9.80665  # NormalG_i:[0, 1]
            whole_info_i[13] /= (4 if plane["Type"] == 1 else 2)  # LeftWeapon_i:[0, 1]
            whole_info_i[14] /= 300000  # Relative_X_i:[-1, 1]
            whole_info_i[15] /= 300000  # Relative_Y_i:[-1, 1]
            whole_info_i[16] /= 13000  # Relative_Z_i:[-1, 1]
            whole_info_i[17] /= 424436  # Relative_dist_i:[0, 1]
            whole_info_i[18] /= math.pi  # Relative_theta_i:[-1, 1]
            whole_info_i[19] /= (math.pi/2)  # Relative_alpha_i:[-1, 1]
            whole_info_i[21] /= 212247  # Center_dist_i:[0, 1]

            myplane_info.append(whole_info_i)
        my_allplane_info = np.concatenate(myplane_info)
        return my_allplane_info

    def get_enemy_allplane_info(self, cur_index, enemy_allplane_info):
        if enemy_allplane_info is not None:
            cur_plane = self.all_plane_map[cur_index]
            for i in range(5, 10):
                plane = self.all_plane_map[i]
                Relative_X_i = np.array([0])
                Relative_Y_i = np.array([0])
                Relative_Z_i = np.array([0])
                Relative_dist_i = np.array([0])
                Relative_theta_i = np.array([0])
                Relative_alpha_i = np.array([0])
                if plane is not None and cur_plane is not None and i != cur_index:
                    Relative_X_i = -np.array([plane["X"] - cur_plane["X"]])
                    Relative_Y_i = np.array([plane["Y"] - cur_plane["Y"]])
                    Relative_Z_i = np.array([plane["Z"] - cur_plane["Z"]])
                    Relative_dist_i = np.array([self._dis(plane, cur_plane)])
                    Relative_theta_i = np.array([self.pi_bound(self.XY2theta(Relative_X_i[0], Relative_Y_i[0]))])
                    Relative_alpha_i = np.array([self.pi_bound(math.asin(Relative_Z_i[0] / (Relative_dist_i[0] + self.EPS)))])

                    # 归一化处理
                    Relative_X_i[0] /= 300000  # Relative_X_i:[-1, 1]
                    Relative_Y_i[0] /= 300000  # Relative_Y_i:[-1, 1]
                    Relative_Z_i[0] /= 13000  # Relative_Z_i:[-1, 1]
                    Relative_dist_i[0] /= 424436  # Relative_dist_i:[0, 1]
                    Relative_theta_i[0] /= math.pi  # Relative_theta_i:[-1, 1]
                    Relative_alpha_i[0] /= (math.pi / 2)  # Relative_alpha_i:[-1, 1]

                enemy_allplane_info[(i-5)*20+12] = Relative_X_i
                enemy_allplane_info[(i-5)*20+13] = Relative_Y_i
                enemy_allplane_info[(i-5)*20+14] = Relative_Z_i
                enemy_allplane_info[(i-5)*20+15] = Relative_dist_i
                enemy_allplane_info[(i-5)*20+16] = Relative_theta_i
                enemy_allplane_info[(i-5)*20+17] = Relative_alpha_i
            return enemy_allplane_info

        enemyplane_info = []
        for i in range(5, 10):
            plane = self.plane_key_info[i]
            Identification_i = np.array([0]) if plane["Identification"] == "红方" else np.array([1])
            Type_i = np.array([0]) if plane["Type"] == 1 else np.array([1])
            Availability_i = np.array([plane["Availability"]])
            X_i = np.array([0])
            Y_i = np.array([0])
            Z_i = np.array([0])
            Heading_i = np.array([0])
            Pitch_i = np.array([0])
            Roll_i = np.array([0])
            Speed_i = np.array([0])
            IsLocked_i = np.array([0])
            LeftWeapon_i = np.array([0])
            Relative_X_i = np.array([0])
            Relative_Y_i = np.array([0])
            Relative_Z_i = np.array([0])
            Relative_dist_i = np.array([0])
            Relative_theta_i = np.array([0])
            Relative_alpha_i = np.array([0])
            Is_man_and_incenter_i = np.array([0])
            Center_dist_i = np.array([212247])
            plane = self.all_plane_map[i]
            cur_plane = self.all_plane_map[cur_index]
            if plane is not None:
                X_i = -np.array([plane["X"]])
                Y_i = np.array([plane["Y"]])
                Z_i = np.array([plane["Z"]])
                Heading_i = -np.array([plane["Heading"]])
                Pitch_i = np.array([plane["Pitch"]])
                Roll_i = -np.array([plane["Roll"]])
                Speed_i = np.array([plane["Speed"]])
                IsLocked_i = np.array([plane["IsLocked"]])
                LeftWeapon_i = np.array([self.enemy_leftweapon[self.all_id_map[i]]])
                if i != cur_index and cur_plane is not None:
                    Relative_X_i = -np.array([plane["X"] - cur_plane["X"]])
                    Relative_Y_i = np.array([plane["Y"] - cur_plane["Y"]])
                    Relative_Z_i = np.array([plane["Z"] - cur_plane["Z"]])
                    Relative_dist_i = np.array([self._dis(plane, cur_plane)])
                    Relative_theta_i = np.array([self.pi_bound(self.XY2theta(Relative_X_i[0], Relative_Y_i[0]))])
                    Relative_alpha_i = np.array([self.pi_bound(math.asin(Relative_Z_i[0] / (Relative_dist_i[0] + self.EPS)))])
                if plane["Type"] == 1 and self.is_in_center(plane):
                    Is_man_and_incenter_i = np.array([1])
                if plane["Type"] == 1:
                    if self.is_in_center(plane):
                        Center_dist_i = np.array([0])
                    else:
                        distance_to_center = (plane['X'] ** 2 + plane["Y"] ** 2 + (plane["Alt"] - 9000) ** 2) ** 0.5
                        Center_dist_i = np.array([distance_to_center - 50000])
            whole_info_i = np.concatenate((Identification_i, Type_i, Availability_i, X_i,
                                           Y_i, Z_i, Heading_i, Pitch_i, Roll_i, Speed_i,
                                           IsLocked_i, LeftWeapon_i,
                                           Relative_X_i, Relative_Y_i, Relative_Z_i, Relative_dist_i,
                                           Relative_theta_i, Relative_alpha_i, Is_man_and_incenter_i,
                                           Center_dist_i))
            # 归一化处理
            plane = self.plane_key_info[i]
            whole_info_i[3] /= 150000  # X_i:[-1, 1]
            whole_info_i[4] /= 150000  # Y_i:[-1, 1]
            whole_info_i[5] = (whole_info_i[5] - 2000) / 13000 if plane["Type"] == 1 else (whole_info_i[5] - 2000) / 8000  # Z_i:[0, 1]
            whole_info_i[6] /= math.pi  # Heading_i:[-1, 1]
            whole_info_i[7] /= (math.pi / 2)  # Pitch_i:[-1, 1]
            whole_info_i[8] /= math.pi  # Roll_i:[-1, 1]
            whole_info_i[9] = (whole_info_i[9] - 150) / 250 if plane["Type"] == 1 else (whole_info_i[9] - 100) / 200  # Speed_i:[0, 1]
            whole_info_i[11] /= (4 if plane["Type"] == 1 else 2)  # LeftWeapon_i:[0, 1]
            whole_info_i[12] /= 300000  # Relative_X_i:[-1, 1]
            whole_info_i[13] /= 300000  # Relative_Y_i:[-1, 1]
            whole_info_i[14] /= 13000  # Relative_Z_i:[-1, 1]
            whole_info_i[15] /= 424436  # Relative_dist_i:[0, 1]
            whole_info_i[16] /= math.pi  # Relative_theta_i:[-1, 1]
            whole_info_i[17] /= (math.pi / 2)  # Relative_alpha_i:[-1, 1]
            whole_info_i[19] /= 212247  # Center_dist_i:[0, 1]

            enemyplane_info.append(whole_info_i)
        enemy_allplane_info = np.concatenate(enemyplane_info)
        return enemy_allplane_info

    def get_all_missiles_info(self, cur_index):
        cur_plane = self.plane_key_info[cur_index]
        my_missiles_info = []
        for i in range(24):
            if i < 12:
                Identification_mi = np.array([0]) if cur_plane["Identification"] == "红方" else np.array([1])
            else:
                Identification_mi = np.array([1]) if cur_plane["Identification"] == "红方" else np.array([0])
            Availability_mi = np.array([0])
            IsLaunched_mi = np.array([0])
            Hit_target_mi = np.array([0])
            X_mi = np.array([0])
            Y_mi = np.array([0])
            Z_mi = np.array([0])
            Heading_mi = np.array([0])
            Pitch_mi = np.array([0])
            Roll_mi = np.array([0])
            Speed_mi = np.array([0])
            Launcher_X_mi = np.array([0])
            Launcher_Y_mi = np.array([0])
            Launcher_Z_mi = np.array([0])
            Launcher_dist_mi = np.array([0])
            Launcher_theta_mi = np.array([0])
            Launcher_alpha_mi = np.array([0])
            Launcher_look_mi = np.array([0])
            EngageTarget_X_mi = np.array([0])
            EngageTarget_Y_mi = np.array([0])
            EngageTarget_Z_mi = np.array([0])
            EngageTarget_dist_mi = np.array([0])
            EngageTarget_theta_mi = np.array([0])
            EngageTarget_alpha_mi = np.array([0])
            Nearest_myplane_X_mi = np.array([0])
            Nearest_myplane_Y_mi = np.array([0])
            Nearest_myplane_Z_mi = np.array([0])
            Nearest_myplane_dist_mi = np.array([0])
            Nearest_myplane_theta_mi = np.array([0])
            Nearest_myplane_alpha_mi = np.array([0])
            Nearest_myplane_look_mi = np.array([0])
            Flight_time_mi = np.array([0])
            Flight_dist_mi = np.array([0])
            Is_terminal_mi = np.array([0])
            Is_slowingdown_mi = np.array([0])
            if self.all_missile_map[i] is not None:
                missile = self.all_missile_map[i]
                Availability_mi = np.array([missile["Availability"]])
                X_mi = -np.array([missile["X"]])
                Y_mi = np.array([missile["Y"]])
                Z_mi = np.array([missile["Z"]])
                Heading_mi = -np.array([missile["Heading"]])
                Pitch_mi = np.array([missile["Pitch"]])
                Roll_mi = -np.array([missile["Roll"]])
                Speed_mi = np.array([missile["Speed"]])
                if i < 12:
                    if missile["LauncherID"] in self.my_allplane_maps:
                        Launcher = self.my_allplane_maps[missile["LauncherID"]]
                    else:
                        Launcher = None
                else:
                    if missile["LauncherID"] in self.enemy_allplane_maps:
                        Launcher = self.enemy_allplane_maps[missile["LauncherID"]]
                    else:
                        Launcher = None
                if Launcher is not None:
                    Launcher_X_mi = -np.array([missile["X"] - Launcher["X"]])
                    Launcher_Y_mi = np.array([missile["Y"] - Launcher["Y"]])
                    Launcher_Z_mi = np.array([missile["Z"] - Launcher["Z"]])
                    Launcher_dist_mi = np.array([self._dis(missile, Launcher)])
                    Launcher_theta_mi = np.array([self.pi_bound(self.XY2theta(Launcher_X_mi[0], Launcher_Y_mi[0]))])
                    Launcher_alpha_mi = np.array([self.pi_bound(math.asin(Launcher_Z_mi[0] / (Launcher_dist_mi[0] + self.EPS)))])
                    Launcher_look_mi = np.array([self.is_looking(Launcher, missile)])
                if i < 12:
                    if missile["EngageTargetID"] in self.enemy_allplane_maps:
                        EngageTarget = self.enemy_allplane_maps[missile["EngageTargetID"]]
                    else:
                        EngageTarget = None
                else:
                    if missile["EngageTargetID"] in self.my_allplane_maps:
                        EngageTarget = self.my_allplane_maps[missile["EngageTargetID"]]
                    else:
                        EngageTarget = None
                if EngageTarget is not None:
                    EngageTarget_X_mi = -np.array([missile["X"] - EngageTarget["X"]])
                    EngageTarget_Y_mi = np.array([missile["Y"] - EngageTarget["Y"]])
                    EngageTarget_Z_mi = np.array([missile["Z"] - EngageTarget["Z"]])
                    EngageTarget_dist_mi = np.array([self._dis(missile, EngageTarget)])
                    EngageTarget_theta_mi = np.array([self.pi_bound(self.XY2theta(EngageTarget_X_mi[0], EngageTarget_Y_mi[0]))])
                    EngageTarget_alpha_mi = np.array([self.pi_bound(math.asin(EngageTarget_Z_mi[0] / (EngageTarget_dist_mi[0] + self.EPS)))])

                Nearest_myplane = self.nearest_myplane(missile, i)
                if Nearest_myplane is not None:
                    Nearest_myplane_X_mi = -np.array([missile["X"] - Nearest_myplane["X"]])
                    Nearest_myplane_Y_mi = np.array([missile["Y"] - Nearest_myplane["Y"]])
                    Nearest_myplane_Z_mi = np.array([missile["Z"] - Nearest_myplane["Z"]])
                    Nearest_myplane_dist_mi = np.array([self._dis(missile, Nearest_myplane)])
                    Nearest_myplane_theta_mi = np.array([self.pi_bound(self.XY2theta(Nearest_myplane_X_mi[0], Nearest_myplane_Y_mi[0]))])
                    Nearest_myplane_alpha_mi = np.array([self.pi_bound(math.asin(Nearest_myplane_Z_mi[0] / (Nearest_myplane_dist_mi[0] + self.EPS)))])
                    Nearest_myplane_look_mi = np.array([self.is_looking(Nearest_myplane, missile)])
                IsLaunched_mi = np.array([1])
                if missile["EngageTargetID"] not in self.enemy_allplane_maps:
                    Hit_target_mi = np.array([1])
                Flight_time_mi = np.array([missile["CurTime"] - self.missile_launch_time[missile["ID"]]])
                Flight_dist_mi = np.array([self.missile_path_length[missile["ID"]]])
                if EngageTarget is not None:
                    Is_terminal_mi = np.array([EngageTarget_dist_mi[0] <= 20000])
                Is_slowingdown_mi = np.array([Flight_time_mi[0] > 30])
            whole_info_i = np.concatenate((Identification_mi, Availability_mi, X_mi,
                                           Y_mi, Z_mi, Heading_mi, Pitch_mi, Roll_mi, Speed_mi,
                                           Launcher_X_mi, Launcher_Y_mi, Launcher_Z_mi,
                                           Launcher_dist_mi, Launcher_theta_mi, Launcher_alpha_mi, Launcher_look_mi,
                                           EngageTarget_X_mi, EngageTarget_Y_mi, EngageTarget_Z_mi,
                                           EngageTarget_dist_mi, EngageTarget_theta_mi, EngageTarget_alpha_mi,
                                           Nearest_myplane_X_mi, Nearest_myplane_Y_mi, Nearest_myplane_Z_mi,
                                           Nearest_myplane_dist_mi, Nearest_myplane_theta_mi, Nearest_myplane_alpha_mi, Nearest_myplane_look_mi,
                                           IsLaunched_mi, Hit_target_mi, Flight_time_mi, Flight_dist_mi, Is_terminal_mi, Is_slowingdown_mi))

            # 归一化处理
            whole_info_i[2] /= 150000  # X_mi:[-1, 1]
            whole_info_i[3] /= 150000  # Y_mi:[-1, 1]
            whole_info_i[4] = (whole_info_i[4] - 2000) / 30000  # Z_mi:[0, 1]
            whole_info_i[5] /= math.pi  # Heading_mi:[-1, 1]
            whole_info_i[6] /= (math.pi/2)  # Pitch_mi:[-1, 1]
            whole_info_i[7] /= math.pi  # Roll_mi:[-1, 1]
            whole_info_i[8] = (whole_info_i[8] - 400) / 1000  # Speed_mi:[0, 1]
            whole_info_i[9] /= 140000  # Launcher_X_mi:[-1, 1]
            whole_info_i[10] /= 140000  # Launcher_Y_mi:[-1, 1]
            whole_info_i[11] = (whole_info_i[11] + 13000) / 41000  # Launcher_Z_mi:[0, 1]
            whole_info_i[12] /= 140000  # Launcher_dist_mi:[0, 1]
            whole_info_i[13] /= math.pi  # Launcher_theta_mi:[-1, 1]
            whole_info_i[14] /= (math.pi/2)  # Launcher_alpha_mi:[-1, 1]
            whole_info_i[16] /= 80000  # EngageTarget_X_mi:[-1, 1]
            whole_info_i[17] /= 80000  # EngageTarget_Y_mi:[-1, 1]
            whole_info_i[18] = (whole_info_i[18] + 13000) / 41000  # EngageTarget_Z_mi:[0, 1]
            whole_info_i[19] /= 80000  # EngageTarget_dist_mi:[0, 1]
            whole_info_i[20] /= math.pi  # EngageTarget_theta_mi:[-1, 1]
            whole_info_i[21] /= (math.pi / 2)  # EngageTarget_alpha_mi:[-1, 1]
            whole_info_i[22] /= 140000  # Nearest_myplane_X_mi:[-1, 1]
            whole_info_i[23] /= 140000  # Nearest_myplane_Y_mi:[-1, 1]
            whole_info_i[24] = (whole_info_i[24] + 13000) / 41000  # Nearest_myplane_Z_mi:[0, 1]
            whole_info_i[25] /= 140000  # Nearest_myplane_dist_mi:[0, 1]
            whole_info_i[26] /= math.pi  # Nearest_myplane_theta_mi:[-1, 1]
            whole_info_i[27] /= (math.pi / 2)  # Nearest_myplane_alpha_mi:[-1, 1]
            whole_info_i[31] /= 120  # Flight_time_mi:[0, 1]
            whole_info_i[32] /= 100000  # Flight_dist_mi:[0, 1]

            my_missiles_info.append(whole_info_i)
        my_allmissiles_info = np.concatenate(my_missiles_info)
        return my_allmissiles_info

    def XY2theta(self, X, Y): #theta:(-pi, pi]
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

    def pi_bound(self, theta): #将角度限制到(-pi,pi]
        while theta > math.pi:
            theta -= 2*math.pi
        while theta <= -math.pi:
            theta += 2*math.pi
        return theta

    def is_in_center(self, plane): #判定是否在中心区域
        distance_to_center = (plane['X']**2 + plane["Y"]**2 + (plane["Alt"] - 9000)**2)**0.5
        if distance_to_center <= 50000 and plane["Alt"] >= 2000 and plane['Alt'] <= 16000:
            return True
        return False

    def is_looking(self, item0, item1): #判定item0是否正监测着item1
        if item0["Type"] == 1:
            launch_range = 80000
        else:
            launch_range = 60000
        looking_theta = math.pi / 3
        if item0['Type'] == 2:
            looking_theta /= 2
        item_distance = self._dis(item0, item1)
        item_theta = self.XY2theta(item1['X'] - item0['X'], item1['Y'] - item0['Y'])
        if item_distance < launch_range and \
                abs(self.pi_bound(item_theta - item0["Heading"])) < looking_theta:
            return 1
        return 0

    def nearest_myplane(self, item, i):
        min_distance = 450000
        close_myplane = None
        my_all_planes = self.my_allplane_infos if i < 12 else self.enemy_allplane_infos
        for myplane in my_all_planes:
            distance = self._dis(item, myplane)  # 计算空间两点距离
            if distance < min_distance:  # 计算出一个离得最近的敌方实体
                min_distance = distance
                close_myplane = myplane
        return close_myplane  # 返回最近的敌方飞机信息

    def process(self, sim_time, obs_side):
        """
        输入当前模拟时间和当前我方态势信息，输出处理好的特征矩阵
        :param sim_time: 模拟时间
        :param obs_side: 我方态势信息
        :return: feature -> (5, 1064) tensor
        """
        self.process_observation(obs_side, sim_time)
        self.process_planemap(sim_time)
        self.process_missilemap(sim_time)
        all_missiles = None
        my_allplane = None
        enemy_allplane = None
        all_info = []
        for cur_index in range(5):
            current_control = self.get_current_control_info(cur_index)
            my_allplane = self.get_my_allplane_info(cur_index, my_allplane)
            enemy_allplane = self.get_enemy_allplane_info(cur_index, enemy_allplane)
            if all_missiles is None:
                all_missiles = self.get_all_missiles_info(cur_index)
            cur_whole_info = np.concatenate((current_control, my_allplane, enemy_allplane, all_missiles))
            all_info.append(cur_whole_info.reshape(1, -1))
        final_state = np.concatenate(all_info)
        return final_state


if __name__ == '__main__':
    feature = Feature()
    obs = np.load('replay_data_1200.npz', allow_pickle=True)
    obs = obs['obs']
    tot_len = len(obs)
    # test_index = -50
    # print(obs[test_index])
    print("tot_len:", tot_len)

    data_list = []
    for i in range(1, tot_len):
        t = i + 1  # 从第2帧开始处理（无法学习初始化布局），到最后1帧结束（包含最后1帧）
        data_list.append(feature.process(t, obs[i]['blue']))
        print(f'step:{i} real_t:{t}')
    print("data_list:", len(data_list))

    # np.savetxt("test_data.txt", data_list[test_index], fmt='%.2f', delimiter=',')
    # test_data = data_list[test_index]
    # mask = np.argwhere(test_data>1)
    # print(mask)
    # for i, j in mask:
    #     print(test_data[i, j])