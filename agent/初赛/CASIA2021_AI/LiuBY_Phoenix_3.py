
from typing import List
from agent.agent import Agent
from env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
import copy
import random
import math


class LiuBY_Phoenix_Agent(Agent):
    def __init__(self, name, config):
        super(LiuBY_Phoenix_Agent, self).__init__(name, config["side"])
        self._init()

    def _init(self):
        self.my_uvas_infos = []
        self.my_manned_info = []
        self.my_allplane_infos = []
        self.my_missile_infos = []
        self.enemy_uvas_infos = []
        self.enemy_manned_info = []
        self.enemy_allplane_infos = []
        self.enemy_missile_infos = []
        self.enemy_leftweapon = {}

        self.attack_handle_enemy = {}

        self.missile_path_length = {}
        self.missile_last_pisition = {}
        self.missile_init_time = {}
        self.attacked_flag = {}
        self.combine_fight = {}
        self.combine_turn = {}

        self.curtime_has_attack = {}
        self.curtime_has_layout = {}
        self.lasttime_has_layout = {}

        self.escape_toward = {}
        self.escape_last_dangerID = {}

        self.missile_to_enemy_manned = 0
        self.missile_to_enemy_uvas = 0

        self.my_score = 0
        self.enemy_score = 0

        self.missile_path_distance = self.get_missile_data()['path_distance']
        self.missile_v = self.get_missile_data()['v']

        self.close = 0
        self.my_allplane_maps = {}

    def reset(self, **kwargs):
        self.my_uvas_infos.clear()
        self.my_manned_info.clear()
        self.my_allplane_infos.clear()
        self.my_missile_infos.clear()
        self.enemy_uvas_infos.clear()
        self.enemy_manned_info.clear()
        self.enemy_allplane_infos.clear()
        self.enemy_missile_infos.clear()
        self.enemy_leftweapon.clear()

        self.attack_handle_enemy.clear()

        self.missile_path_length.clear()
        self.missile_last_pisition.clear()
        self.missile_init_time.clear()
        self.attacked_flag.clear()
        self.combine_fight.clear()
        self.combine_turn.clear()

        self.curtime_has_attack.clear()
        self.curtime_has_layout.clear()
        self.lasttime_has_layout.clear()

        self.escape_toward.clear()
        self.escape_last_dangerID.clear()

        self.missile_to_enemy_manned = 0
        self.missile_to_enemy_uvas = 0

        self.my_score = 0
        self.enemy_score = 0

        self.close = 0
        self.my_allplane_maps = {}

        pass

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:

        cmd_list = []

        self.process_decision(sim_time, obs_side, cmd_list)

        return cmd_list

    def process_decision(self, sim_time, obs_side, cmd_list):

        self.process_observation(obs_side)

        if sim_time == 1:
            self.init_pos(cmd_list)

        if sim_time >= 2:
            self.process_move(sim_time, cmd_list)
            self.process_attack(sim_time, cmd_list)

    def process_observation(self, obs_side):

        my_entity_infos = obs_side['platforminfos']
        if len(my_entity_infos) < 1:
            return
        my_manned_info = []
        my_uvas_infos = []
        my_allplane_infos = []
        for uvas_info in my_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001:
                uvas_info["Z"] = uvas_info["Alt"]
                if uvas_info['Type'] == 1:
                    my_manned_info.append(uvas_info)
                if uvas_info['Type'] == 2:
                    my_uvas_infos.append(uvas_info)
                my_allplane_infos.append(uvas_info)

        if len(my_manned_info) < 1:
            return

        enemy_entity_infos = obs_side['trackinfos']
        enemy_manned_info = []
        enemy_uvas_infos = []
        enemy_allplane_infos = []
        for uvas_info in enemy_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001:
                uvas_info['Z'] = uvas_info['Alt']
                if uvas_info['Type'] == 1:
                    enemy_manned_info.append(uvas_info)
                if uvas_info['Type'] == 2:
                    enemy_uvas_infos.append(uvas_info)
                enemy_allplane_infos.append(uvas_info)

        my_allplane_maps = {}
        for input_entity in my_allplane_infos:
            my_allplane_maps[int(input_entity['ID'])] = input_entity
            self.my_allplane_maps[int(input_entity['ID'])] = input_entity

        missile_infos = obs_side['missileinfos']

        my_missile_infos = []
        enemy_missile_infos = []

        for missile_info in missile_infos:
            if (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001):
                if missile_info['LauncherID'] in self.my_allplane_maps:
                    missile_info["Z"] = missile_info["Alt"]
                    my_missile_infos.append(missile_info)
                else:
                    missile_info["Z"] = missile_info["Alt"]
                    enemy_missile_infos.append(missile_info)

        for enemy_plane in enemy_allplane_infos:
            if enemy_plane['ID'] not in self.enemy_leftweapon:
                if enemy_plane['Type'] == 1:
                    self.enemy_leftweapon[enemy_plane['ID']] = 4
                else:
                    self.enemy_leftweapon[enemy_plane['ID']] = 2

        for missile in my_missile_infos + enemy_missile_infos:
            old_missile_ID_list = []
            for old_missile in self.my_missile_infos + self.enemy_missile_infos:
                old_missile_ID_list.append(old_missile['ID'])
            if missile['ID'] not in old_missile_ID_list:
                self.missile_path_length[missile['ID']] = 0
                self.missile_last_pisition[missile['ID']] = {'X': 0, 'Y': 0, 'Z': 0}
                self.missile_init_time[missile['ID']] = missile['CurTime']
                if missile['LauncherID'] in self.enemy_leftweapon:
                    self.enemy_leftweapon[missile['LauncherID']] -= 1
            else:
                self.missile_path_length[missile['ID']] += TSVector3.distance(
                    self.missile_last_pisition[missile['ID']], missile)
            self.missile_last_pisition[missile['ID']]['X'] = missile['X']
            self.missile_last_pisition[missile['ID']]['Y'] = missile['Y']
            self.missile_last_pisition[missile['ID']]['Z'] = missile['Z']



        for plane in my_allplane_infos + enemy_allplane_infos:

            if plane['ID'] not in self.attacked_flag:
                self.attacked_flag[plane['ID']] = []

            availability_missile_ID_list = []

            for missile in my_missile_infos + enemy_missile_infos:
                availability_missile_ID_list.append(missile['ID'])
                if plane['ID'] == missile['EngageTargetID'] and missile['ID'] not in self.attacked_flag[plane['ID']]:
                    self.attacked_flag[plane['ID']].append(missile['ID'])

            remove_num = 0
            temp_under = range(len(self.attacked_flag[plane['ID']]))
            for i in temp_under:
                missile_ID = self.attacked_flag[plane['ID']][i-remove_num]
                if missile_ID not in availability_missile_ID_list:
                    self.attacked_flag[plane['ID']].remove(missile_ID)
                    remove_num += 1

        for my_plane in my_allplane_infos:
            if my_plane['ID'] not in self.escape_toward:
                self.escape_toward[my_plane['ID']] = 0
            if my_plane['ID'] not in self.escape_last_dangerID:
                self.escape_last_dangerID[my_plane['ID']] = 0

        self.curtime_has_attack = {}
        for enemy in enemy_allplane_infos:
            self.curtime_has_attack[enemy['ID']] = 0

        if self.is_in_center(my_manned_info[0]):
            self.my_score += 1
        if self.is_in_center(enemy_manned_info[0]):
            self.enemy_score += 1

        self.my_uvas_infos = my_uvas_infos
        self.my_manned_info = my_manned_info
        self.my_allplane_infos = my_allplane_infos
        self.my_missile_infos = my_missile_infos
        self.enemy_uvas_infos = enemy_uvas_infos
        self.enemy_manned_info = enemy_manned_info
        self.enemy_allplane_infos = enemy_allplane_infos
        self.enemy_missile_infos = enemy_missile_infos
        self.curtime_has_layout = {}


        combine_distance = 15000
        if len(self.combine_fight) == 0:

            for enemy in self.enemy_manned_info + self.enemy_uvas_infos:
                if self.missile_to_enemy_manned >= 1 and enemy['Type'] == 1 or \
                        self.missile_to_enemy_uvas - (4 - len(self.enemy_uvas_infos)) >= 1:
                    can_attack_num = 0
                    temp_list = []
                    for can_attack_member in self.my_uvas_infos:
                        if TSVector3.distance(can_attack_member, enemy) < combine_distance:
                            if can_attack_member['LeftWeapon'] > 0:
                                can_attack_num += can_attack_member['LeftWeapon']
                                temp_list.append(can_attack_member)
                    combine_flag = 0
                    if len(temp_list) >= 2:
                        for i in temp_list:
                            for j in temp_list:
                                if i['ID'] != j['ID']:
                                    theta_i = self.XY2theta(i['X']-enemy['X'], i['Y']-enemy['Y'])
                                    theta_j = self.XY2theta(j['X']-enemy['X'], j['Y']-enemy['Y'])
                                    theta_ij = abs(self.pi_bound(theta_i-theta_j))
                                    if theta_ij >= math.pi/2:
                                        combine_flag = 1
                                        break
                            if combine_flag == 1:
                                break

                    if can_attack_num >= 3 and combine_flag == 1:
                        for temp in temp_list:
                            self.combine_fight[temp['ID']] = enemy['ID']
                        self.combine_turn[enemy['ID']] = False
        else:
            for my_plane in self.my_uvas_infos:
                enemy_plane = self.get_plane_by_id(list(self.combine_fight.values())[0])
                if enemy_plane == None or self.availability(enemy_plane) == False:
                    break
                if self.availability(my_plane) and TSVector3.distance(my_plane, enemy_plane) < combine_distance and\
                        my_plane['ID'] not in self.combine_fight:
                    self.combine_fight[my_plane['ID']] = enemy_plane['ID']
            for my_plane_ID in list(self.combine_fight.keys()):
                my_plane = self.get_plane_by_id(my_plane_ID)
                if my_plane == None or self.availability(my_plane) == False:
                    del self.combine_fight[my_plane_ID]
                elif my_plane['LeftWeapon'] <= 0:
                    del self.combine_fight[my_plane_ID]
                else:
                    enemy_plane_ID = self.combine_fight[my_plane_ID]
                    enemy_plane = self.get_plane_by_id(enemy_plane_ID)
                    if enemy_plane == None or self.availability(enemy_plane) == False:
                        del self.combine_fight[my_plane_ID]
                        if enemy_plane_ID in self.combine_turn:
                            del self.combine_turn[enemy_plane_ID]

        curtime_combine_turn = {}
        for enemy_plane_ID in self.combine_turn:
            if self.combine_turn[enemy_plane_ID] == False:
                curtime_combine_turn[enemy_plane_ID] = True
        for my_plane_ID in self.combine_fight:
            my_plane = self.get_plane_by_id(my_plane_ID)
            enemy_plane_ID = self.combine_fight[my_plane_ID]
            if self.combine_turn[enemy_plane_ID] == True or curtime_combine_turn[enemy_plane_ID] == False:
                continue
            enemy_plane = self.get_plane_by_id(enemy_plane_ID)
            if self.is_looking(my_plane, enemy_plane, my_plane['Heading']) == False:
                curtime_combine_turn[enemy_plane_ID] = False
        for enemy_plane_ID in curtime_combine_turn:
            if curtime_combine_turn[enemy_plane_ID]:
                self.combine_turn[enemy_plane_ID] = True

    def init_pos(self, cmd_list):

        leader_original_pos = {}
        leader_heading = 0
        if self.name == "red":
            leader_original_pos = {"X": -125000, "Y": 0, "Z": 9000}
            leader_heading = 90
        else :
            leader_original_pos = {"X": 125000, "Y": 0, "Z": 9000}
            leader_heading = 270

        interval_distance = 5000
        for leader in self.my_manned_info:
            cmd_list.append(CmdEnv.make_entityinitinfo(leader['ID'], leader_original_pos['X'], leader_original_pos['Y'], leader_original_pos['Z'],400, leader_heading))

        sub_index = 0
        for sub in self.my_uvas_infos:
            sub_pos = copy.deepcopy(leader_original_pos)
            if sub_index & 1 == 0:
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] + interval_distance, sub_pos['Z'],300 * 0.6, leader_heading))
            else:
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] - interval_distance, sub_pos['Z'],300 * 0.6, leader_heading))
                interval_distance *= 2
            sub_index += 1


    def process_attack(self, sim_time, cmd_list):
        """
        处理攻击，根据敌方信息进行攻击
        :param sim_time: 当前想定已经运行时间
        :param cmd_list保存所有的决策完毕任务指令列表
        """
        combine_distance = 15000
        for plane in self.my_manned_info + self.my_uvas_infos:
            if len(self.my_manned_info) > 0 and len(self.enemy_manned_info) > 0 and plane['LeftWeapon'] > 0:

                true_attack_enemy = {}
                my_leader = self.my_manned_info[0]
                data = self.get_move_data(plane)
                for enemy_plane in self.enemy_allplane_infos:
                    if TSVector3.distance(enemy_plane, my_leader) < 15000 and len(self.attacked_flag[enemy_plane['ID']])+self.curtime_has_attack[enemy_plane['ID']] <= 1:
                        if plane['Type'] == 1 and plane['LeftWeapon'] > 1 and self.is_attacking(plane, enemy_plane) is False:
                            cmd_list.append(CmdEnv.make_attackparam(plane['ID'], enemy_plane['ID'],
                                                                    5000 / data['launch_range']))
                            true_attack_enemy = enemy_plane
                        elif plane['Type'] == 2 and plane['LeftWeapon'] > 0:
                            min_flag = True
                            for my_uvas in self.my_uvas_infos:
                                if my_uvas['LeftWeapon'] > 0 and \
                                        TSVector3.distance(my_uvas, enemy_plane) < TSVector3.distance(plane, enemy_plane):
                                    min_flag = False
                            if min_flag and TSVector3.distance(enemy_plane, plane) < data['launch_range']:
                                if self.is_looking(plane, enemy_plane, plane['Heading']):
                                    cmd_list.append(CmdEnv.make_attackparam(plane['ID'], enemy_plane['ID'], 1))
                                    true_attack_enemy = enemy_plane
                                else:
                                    cmd_list.append(
                                        CmdEnv.make_followparam(plane['ID'], enemy_plane['ID'],
                                                                data['move_max_speed'],
                                                                data['move_max_acc'], data['move_max_g']))

                fire_flag = 1
                danger_tangent_distance = 30000
                for danger in self.enemy_allplane_infos + self.enemy_missile_infos:
                    if self.availability(danger) and TSVector3.distance(danger, plane) < danger_tangent_distance and \
                            self.get_tangent_shot_time(danger, plane) <= 2:
                        fire_flag = 0
                        break
                if fire_flag == 0:
                    continue
                attack_enemy_list = self.attack_select(plane)
                for attack_enemy in attack_enemy_list:

                    attack_distance = self.get_attack_area(plane, attack_enemy)

                    if plane['Type'] == 1:

                        if plane['LeftWeapon'] <= 1:
                            if TSVector3.distance(plane, self.enemy_manned_info[0]) < 3000:
                                self.close += 1
                            elif TSVector3.distance(plane, self.enemy_manned_info[0]) < 5000 and \
                                    abs(self.pi_bound(plane['Heading']-self.enemy_manned_info[0]['Heading'])) < math.pi / 18:
                                self.close += 1
                            else:
                                self.close = 0
                            if self.close > 5:
                                if random.randint(0, 10) < 5:
                                    leader_fire_route_list = [{"X": plane['X'] + 10000*math.sin(plane['Heading']),
                                                               "Y": plane['Y'] + 10000*math.cos(plane['Heading']),
                                                               "Z": data['area_max_alt']}, ]  
                                    cmd_list.append(
                                        CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                    data['move_max_speed'], data['move_max_acc'],
                                                                    data["move_max_g"]));
                                elif abs(self.pi_bound(plane['Heading']-self.enemy_manned_info[0]['Heading'])) < math.pi / 9:
                                    cmd_list.append(
                                        CmdEnv.make_attackparam(plane['ID'], self.enemy_manned_info[0]['ID'], 1))
                                    true_attack_enemy = self.enemy_manned_info[0]

                            break
                        if self.is_attacking(plane, attack_enemy) == False and\
                                TSVector3.distance(attack_enemy, plane) < attack_distance and\
                                self.is_looking(plane, attack_enemy, plane['Heading']):
                            cmd_list.append(CmdEnv.make_attackparam(plane['ID'], attack_enemy['ID'],
                                                                    attack_distance / data['launch_range']))
                            true_attack_enemy = attack_enemy
                            break

                    else:

                        if attack_enemy['Type'] == 1 and (self.missile_to_enemy_manned >= 1 or \
                                self.missile_to_enemy_uvas - (4 - len(self.enemy_uvas_infos)) >= 1 or plane['LeftWeapon'] <= 1):
                            attack_distance = combine_distance

                        if attack_distance == combine_distance:
                            if plane['ID'] in self.combine_fight and attack_enemy['ID'] == self.combine_fight[plane['ID']]:
                                
                                fire_flag = 1
                                
                                for missile in self.my_missile_infos:
                                    if self.availability(missile) and missile['LauncherID'] == plane['ID'] and\
                                            missile['EngageTargetID'] == attack_enemy['ID'] and \
                                            missile['CurTime']-self.missile_init_time[missile['ID']] <= 10:
                                        fire_flag = 0
                                        break

                                close_missile = {}
                                min_distance = 450000
                                for missile in self.my_missile_infos:
                                    if self.availability(missile) and missile['EngageTargetID'] == attack_enemy['ID']:
                                        distance = TSVector3.distance(missile, attack_enemy)
                                        if distance < min_distance:
                                            min_distance = distance
                                            close_missile = missile

                                far_plane_ID = -1
                                max_distance = -1
                                for my_combine_plane_ID in self.combine_fight:
                                    my_combine_plane = self.get_plane_by_id(my_combine_plane_ID)
                                    if attack_enemy['ID'] == self.combine_fight[my_combine_plane_ID]:
                                        distance = TSVector3.distance(my_combine_plane, attack_enemy)
                                        if distance > max_distance:
                                            max_distance = distance
                                            far_plane_ID = my_combine_plane_ID

                                if len(close_missile) == 0:
                                    if far_plane_ID != plane['ID']:
                                        fire_flag = 0
                                else:
                                    if min_distance - 1000 > TSVector3.distance(plane, attack_enemy):
                                        fire_flag = 0

                                if plane['LeftWeapon'] > 0 and fire_flag == 1 and \
                                        TSVector3.distance(attack_enemy, plane) < data['launch_range']:
                                    cmd_list.append(CmdEnv.make_attackparam(plane['ID'], attack_enemy['ID'],
                                                                            attack_distance / data['launch_range']))
                                    true_attack_enemy = attack_enemy
                                    break

                        else:
                            if plane['LeftWeapon'] > 0 and self.is_attacking(plane, attack_enemy) == False and \
                                    len(self.attacked_flag[attack_enemy['ID']])+self.curtime_has_attack[attack_enemy['ID']] <= 1 and \
                                    TSVector3.distance(attack_enemy, plane) < attack_distance and \
                                    self.is_looking(plane, attack_enemy, plane['Heading']):
                                cmd_list.append(CmdEnv.make_attackparam(plane['ID'], attack_enemy['ID'],
                                                                        attack_distance / data['launch_range']))
                                true_attack_enemy = attack_enemy
                                break

                if len(true_attack_enemy) > 0:
                    if true_attack_enemy['Type'] == 1:
                        self.missile_to_enemy_manned += 1
                    else:
                        self.missile_to_enemy_uvas += 1
                    self.curtime_has_attack[true_attack_enemy['ID']] += 1


    def process_move(self, sim_time, cmd_list):
        for plane in self.my_allplane_infos:
            self.single_process_move(sim_time, cmd_list, plane)

        self.lasttime_has_layout = self.curtime_has_layout.copy()


    def single_process_move(self, sim_time, cmd_list, plane):

        if len(self.my_manned_info) > 0:

            dest_distance = 10000
            max_bound = 145000
            enemy_manned = self.enemy_manned_info[0]
            if plane['Type'] == 1 and (self.my_score < self.enemy_score + 10 or enemy_manned['X']**2 + enemy_manned['Y']**2 < 55000**2) and\
                    self.is_in_center(plane):
                max_bound = 45000
            danger_tangent_distance = 30000
            danger_uvas_tangent_distance = danger_tangent_distance
            free_distance = 20000

            data = self.get_move_data(plane)

            danger_enemy_distance = 65000
            very_danger_enemy_distance = 55100

            lock_min_danger = {}
            danger_list = []
            lock_min_distacne = 450000
            lock_min_time = 2000
            for missile in self.enemy_missile_infos:
                if self.availability(missile) and missile['EngageTargetID'] == plane['ID']:
                    distance = TSVector3.distance(plane, missile)
                    if distance > (120 + self.missile_init_time[missile['ID']] - missile['CurTime']) * \
                            (self.get_danger_speed(missile) + data['move_max_speed']):
                        continue
                    if distance > danger_uvas_tangent_distance and plane['Type'] == 2:
                        continue
                    missile_time = self.get_tangent_shot_time(missile, plane)
                    danger_list.append(missile)
                    if missile_time < lock_min_time:
                        lock_min_distacne = distance
                        lock_min_time = missile_time
                        lock_min_danger = missile
            locking_list = []
            locking_min_enemy = {}
            locking_min_time = 2000
            for my_missile in self.my_missile_infos:
                if self.availability(my_missile) and my_missile['LauncherID'] == plane['ID']:
                    enemy_plane = self.get_plane_by_id(my_missile['EngageTargetID'])
                    if enemy_plane != None and self.availability(enemy_plane) and \
                            TSVector3.distance(my_missile, enemy_plane) >= free_distance:
                        locking_list.append(enemy_plane)
                        data_enemy = self.get_move_data(enemy_plane)
                        locking_time = self.get_danger_shot_time(my_missile, self.get_danger_speed(my_missile),
                                                                 enemy_plane, data_enemy['move_max_speed'],
                                                                 enemy_plane['Heading'])
                        if locking_time < locking_min_time:
                            locking_min_enemy = enemy_plane
                            locking_min_time = locking_time

            surround_list = danger_list + []
            for enemy_surround in self.enemy_allplane_infos:
                if self.availability(enemy_surround) and self.enemy_leftweapon[enemy_surround['ID']] > 0:
                    distance = TSVector3.distance(plane, enemy_surround)
                    if distance >= danger_enemy_distance and distance < (1 + 3 ** 0.5) * danger_enemy_distance:
                        surround_list.append(enemy_surround)
            s_dict = self.surround_dict(plane, surround_list)
            surround_flag = 1
            if len(surround_list) < 3:
                surround_flag = 0
            else:
                for enemy_danger_ID in s_dict.keys():
                    if enemy_danger_ID == plane['ID']:
                        continue
                    if s_dict[plane['ID']][enemy_danger_ID]['distance'] != -1:
                        surround_flag = 0
                        break

            if len(danger_list) > 0:

                dx_ = plane['X'] - lock_min_danger['X']
                dy_ = plane['Y'] - lock_min_danger['Y']
                r = (dx_ * dx_ + dy_ * dy_) ** 0.5
                if r < 0.0001:
                    r = 0.0001
                danger_v = self.get_danger_speed(lock_min_danger)

                escape_flag = 'tangent'


                if lock_min_distacne > danger_tangent_distance:
                    self.escape_toward[plane['ID']] = 0
                    if len(danger_list) == 1:
                        escape_flag = 'single_back'
                    else:
                        escape_flag = 'mult_back'

                dx = 0
                dy = 0
                if escape_flag == 'tangent':

                    dx = -dy_ * dest_distance / r
                    dy = dx_ * dest_distance / r

                    if self.escape_last_dangerID[plane['ID']] != lock_min_danger['ID']:
                        self.escape_last_dangerID[plane['ID']] = lock_min_danger['ID']
                        if lock_min_time < 15:
                            self.escape_toward[plane['ID']] = 0

                    if self.escape_toward[plane['ID']] == -1:
                        dx = -dx
                        dy = -dy
                    elif self.escape_toward[plane['ID']] == 0:
                        if dx * plane['X'] + dy * plane['Y'] >= 0 and \
                                ((plane['X'] ** 2 + plane['Y'] ** 2) ** 0.5 >= max_bound - dest_distance and plane[
                                    'Type'] == 1 or \
                                 (plane['X'] ** 2 + plane['Y'] ** 2) ** 0.5 >= max_bound and plane['Type'] == 2):
                            dx = -dx
                            dy = -dy
                            self.escape_toward[plane['ID']] = -1
                        elif plane['Type'] == 2 and \
                                abs(self.pi_bound(
                                    plane['Heading'] - self.XY2theta(dx, dy))) > math.pi / 2 - math.pi / 18 and \
                                abs(self.pi_bound(
                                    plane['Heading'] - self.XY2theta(dx, dy))) < math.pi / 2 + math.pi / 18:
                            turn_flag = random.randint(0, 1)
                            if turn_flag == 1:
                                dx = -dx
                                dy = -dy
                                self.escape_toward[plane['ID']] = -1
                            else:
                                self.escape_toward[plane['ID']] = 1
                        elif dx * math.sin(plane['Heading']) + dy * math.cos(plane['Heading']) < 0:
                            dx = -dx
                            dy = -dy
                            self.escape_toward[plane['ID']] = -1
                        else:
                            self.escape_toward[plane['ID']] = 1

                    if plane['Type'] == 2:
                        dx = -dx_ * dest_distance / r
                        dy = -dy_ * dest_distance / r




                else:
                    if escape_flag == 'single_back':
                        dx = -dx_ * dest_distance / r
                        dy = -dy_ * dest_distance / r

                    else:
                        heading_cut = 100
                        min_turn = 2 * math.pi / heading_cut
                        best_turn_theta = 0
                        best_attacked_time = 1200
                        for i in range(heading_cut):
                            escape_theta = self.pi_bound(i * min_turn)
                            danger_min_attack_t = 2000
                            for danger_item in danger_list:
                                danger_attack_t = self.get_danger_shot_time(danger_item,
                                                                            self.get_danger_speed(danger_item),
                                                                            plane, plane['Speed'], escape_theta)


                                if danger_attack_t < danger_min_attack_t:
                                    danger_min_attack_t = danger_attack_t
                            if danger_min_attack_t < best_attacked_time:
                                best_turn_theta = escape_theta
                                best_attacked_time = danger_min_attack_t


                        dx = math.sin(best_turn_theta) * dest_distance
                        dy = math.cos(best_turn_theta) * dest_distance

                if escape_flag == 'tangent' and lock_min_danger['Type'] != 3 and self.is_looking(lock_min_danger, plane, lock_min_danger[
                    'Heading']) == False:
                    dx = dx_ * dest_distance / r
                    dy = dy_ * dest_distance / r

                dest_x = dx + plane['X']
                dest_y = dy + plane['Y']

                if surround_flag == 1 and escape_flag != 'tangent':
                    surround_min_distance = 450000
                    surround_dx = 0
                    surround_dy = 0
                    closer_plane = {}
                    for plane0 in surround_list:
                        for plane1 in surround_list:
                            if plane0['ID'] != plane1['ID'] and s_dict[plane0['ID']][plane1['ID']][
                                'distance'] != -1 and s_dict[plane0['ID']][plane1['ID']][
                                'distance'] < surround_min_distance:
                                surround_min_distance = s_dict[plane0['ID']][plane1['ID']]['distance']
                                surround_angle = self.XY2theta(s_dict[plane0['ID']][plane1['ID']]['dx'],
                                                               s_dict[plane0['ID']][plane1['ID']]['dy'])
                                distance0 = TSVector3.distance(plane0, plane)
                                distance1 = TSVector3.distance(plane1, plane)
                                if distance0 > distance1:
                                    plane0_angle = self.XY2theta(plane0['X'] - plane['X'], plane0['Y'] - plane['Y'])
                                    surround_angle = self.XY2theta(
                                        0.6 * math.sin(surround_angle) + 0.4 * math.sin(plane0_angle),
                                        0.6 * math.cos(surround_angle) + 0.4 * math.cos(plane0_angle))
                                else:
                                    plane1_angle = self.XY2theta(plane1['X'] - plane['X'], plane1['Y'] - plane['Y'])
                                    surround_angle = self.XY2theta(
                                        0.6 * math.sin(surround_angle) + 0.4 * math.sin(plane1_angle),
                                        0.6 * math.cos(surround_angle) + 0.4 * math.cos(plane1_angle))
                                surround_dx = math.sin(surround_angle) * dest_distance
                                surround_dy = math.cos(surround_angle) * dest_distance

                    surround_dest_x = surround_dx + plane['X']
                    surround_dest_y = surround_dy + plane['Y']
                    dest_x = surround_dest_x
                    dest_y = surround_dest_y

                dest = self.dest_bound(plane, dest_x, dest_y, max_bound, dest_distance)

                if plane['Type'] == 2 and escape_flag == 'tangent':
                    dest = self.rect_bound ({'X':dest_x, 'Y':dest_y}, max_bound)

                dest_x = dest['X']
                dest_y = dest['Y']

                head_x = math.sin(plane['Heading'])
                head_y = math.cos(plane['Heading'])
                leader_fire_route_list = [{"X": dest_x, "Y": dest_y,
                                           "Z": data['area_max_alt'] - 500}, ]

                if escape_flag == 'tangent' or head_x * (dest_x - plane['X']) + head_y * (dest_y - plane['Y']) > 0:
                    cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                data['move_max_speed'], data['move_max_acc'],
                                                                data["move_max_g"]));
                else:
                    cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                data['move_min_speed'], data['move_max_acc'],
                                                                data["move_max_g"]));
            elif len(locking_list) > 0:

                self.escape_toward[plane['ID']] = 0

                only_track_flag = 0

                myplane_enemy_theta = self.XY2theta(locking_min_enemy['X']-plane['X'], locking_min_enemy['Y']-plane['Y'])
                track_theta = self.pi_bound(plane['Heading'] - myplane_enemy_theta)
                enemy_theta = self.pi_bound(locking_min_enemy['Heading'] - myplane_enemy_theta)

                myplane_enemy_distance = TSVector3.distance(plane, locking_min_enemy)

                enemy_v_tangent = locking_min_enemy['Speed'] * math.sin(enemy_theta)
                enemy_w_tangent = enemy_v_tangent / myplane_enemy_distance
                enemy_v_back = locking_min_enemy['Speed'] * math.cos(enemy_theta)
                my_w_tangent = data['move_max_g'] * 9.8 / plane['Speed']

                looking_theta = math.pi/3
                if plane['Type'] == 2:
                    looking_theta /= 2

                if abs(track_theta) >= looking_theta * 0.9:
                    only_track_flag = 1
                if abs(self.pi_bound(track_theta - self.pi_bound(enemy_w_tangent * 5))) >= looking_theta * 0.9:
                    only_track_flag = 1

                enemy_distance = TSVector3.distance(locking_min_enemy, plane)
                if enemy_distance >= data['launch_range'] * 0.9:
                    only_track_flag = 1
                if enemy_distance + enemy_v_back * 5 >= data['launch_range'] * 0.9 or \
                        enemy_distance + enemy_v_back * 5 <= data['launch_range'] * 0.1:
                    only_track_flag = 1

                if my_w_tangent * 0.8 <= abs(enemy_w_tangent):
                    only_track_flag = 1

                if locking_min_enemy['Type'] == 1 and plane['Type'] == 2:
                    only_track_flag = 1

                if self.is_out_of_map(plane['X'], plane['Y']):
                    only_track_flag = 1

                if only_track_flag == 1:
                    cmd_list.append(CmdEnv.make_followparam(plane['ID'], locking_min_enemy['ID'],
                                                data['move_max_speed'],
                                                data['move_max_acc'], data['move_max_g']))
                else:
                    heading_cut = 100
                    min_turn = looking_theta / heading_cut
                    best_turn_theta = -math.pi
                    best_turn_enemy_num = -1
                    for i in range(heading_cut):
                        for j in [-1, 1]:
                            turn_theta = myplane_enemy_theta + i * j * min_turn
                            turn_enemy_num = 0
                            for locking_enemy in locking_list:
                                le_distance = TSVector3.distance(locking_enemy, plane)
                                le_theta = self.XY2theta(locking_enemy['X']-plane['X'],locking_enemy['Y']-plane['Y'])
                                if le_distance < data['launch_range'] and\
                                        abs(self.pi_bound(le_theta - turn_theta)) < looking_theta:
                                    turn_enemy_num += 1
                            if turn_enemy_num > best_turn_enemy_num:
                                best_turn_theta = turn_theta
                                best_turn_enemy_num = turn_enemy_num

                    dx = math.sin(best_turn_theta) * dest_distance
                    dy = math.cos(best_turn_theta) * dest_distance

                    dest_x = plane['X'] + dx
                    dest_y = plane['Y'] + dy

                    leader_fire_route_list = [{"X": dest_x, "Y": dest_y,
                                               "Z": data['area_max_alt']-500}, ]

                    if plane['Type'] == 1:
                        cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                    data['move_max_speed'], data['move_max_acc'],
                                                                    data["move_max_g"]));
                    else:
                        my_v = enemy_v_back
                        if my_v < data['move_min_speed']:
                            my_v = data['move_min_speed']
                        if my_v > data['move_max_speed']:
                            my_v = data['move_max_speed']
                        cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                    my_v, data['move_max_acc'],
                                                                    data["move_max_g"]));

            elif plane['ID'] in self.combine_fight:
                self.escape_toward[plane['ID']] = 0
                cmd_list.append(CmdEnv.make_followparam(plane['ID'], self.combine_fight[plane['ID']], data['move_max_speed'],
                                                        data['move_max_acc'], data['move_max_g']))

            elif surround_flag == 1 and plane['Type'] == 1:

                self.escape_toward[plane['ID']] = 0

                dest_x = 0
                dest_y = 0
                if plane['Type'] == 1:
                    surround_min_distance = 450000
                    dx = 0
                    dy = 0
                    for plane0 in surround_list:
                        for plane1 in surround_list:
                            if plane0['ID'] != plane1['ID'] and s_dict[plane0['ID']][plane1['ID']][
                                'distance'] != -1 and s_dict[plane0['ID']][plane1['ID']][
                                'distance'] < surround_min_distance:
                                surround_min_distance = s_dict[plane0['ID']][plane1['ID']]['distance']
                                dx = s_dict[plane0['ID']][plane1['ID']]['dx'] * dest_distance
                                dy = s_dict[plane0['ID']][plane1['ID']]['dy'] * dest_distance

                    dest_x = dx + plane['X']
                    dest_y = dy + plane['Y']

                    dest = self.dest_bound(plane, dest_x, dest_y, max_bound, dest_distance)
                    dest_x = dest['X']
                    dest_y = dest['Y']

                leader_fire_route_list = [{"X": dest_x, "Y": dest_y,
                                           "Z": data['area_max_alt']-500}, ]

                cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                data['move_max_speed'], data['move_max_acc'],
                                                                data["move_max_g"]));


            else:

                self.escape_toward[plane['ID']] = 0
                if len(self.enemy_manned_info) > 0:
                    enemy_leader = self.enemy_manned_info[0]
                    dest = self.rect_bound(self.layout_point_select(plane), 145000)
                    leader_fire_route_list = [{"X": dest['X'], "Y": dest['Y'], "Z": dest['Z']}, ]
                    cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                data['move_max_speed'], data['move_max_acc'],
                                                                data["move_max_g"]));
                else:
                    cmd_list.append(
                        CmdEnv.make_areapatrolparam(plane['ID'], plane['X'],
                                                    plane['Y'], data['area_max_alt']-500, 200, 100,
                                                    data['move_max_speed'], data['move_max_acc'], data['move_max_g']))


    def is_locked(self, myself):
        locked_flag = False
        for enemy_missile in self.enemy_missile_infos:
            if enemy_missile['EngageTargetID'] == myself['ID']:
                return True
        return False


    def layout_point_select(self, myself):

        data = self.get_move_data(myself)

        isblue = 1
        if self.name == 'red':
            isblue = -1

        if len(self.enemy_manned_info) == 0:
            return {'X': isblue * 40000 * (1 - myself['CurTime'] / (20 * 60)), 'Y': 0, 'Z': data['area_max_alt']-500}
        select_plane = self.enemy_manned_info[0]
        data1 = self.get_move_data(select_plane)
        self.curtime_has_layout[myself['ID']] = select_plane['ID']

        my_Alt = select_plane['Alt']
        if my_Alt >= data['area_max_alt']-500:
            my_Alt = data['area_max_alt']-500

        turn_distance = data1['launch_range'] + 10000
        keep_distance = 30000 + 2000
        goal_R1 = data1['launch_range']
        goal_R2 = 30000

        my_distance = TSVector3.distance(select_plane, myself)




        if my_distance <= keep_distance:
            dest_distance = 10000
            if my_distance > 10000:
                return {'X': select_plane['X'], 'Y': select_plane['Y'], 'Z': data['area_max_alt']-500}
            else:
                enemy2myself_theta = self.XY2theta(myself['X']-select_plane['X'], myself['Y']-select_plane['Y'])
                if abs(self.pi_bound(enemy2myself_theta-select_plane['Heading'])) < math.pi/2:
                    return {'X': myself['X']+dest_distance*math.sin(select_plane['Heading']),
                            'Y': myself['Y']+dest_distance*math.cos(select_plane['Heading']), 'Z': data['area_max_alt']-500}
                else:
                    return {'X': select_plane['X'] + dest_distance * math.sin(select_plane['Heading']),
                            'Y': select_plane['Y'] + dest_distance * math.cos(select_plane['Heading']), 'Z': data['area_max_alt']-500}


        left_num = 0
        right_num = 0
        my_theta = self.XY2theta(select_plane['X']-myself['X'], select_plane['Y']-myself['Y'])
        for plane in self.my_allplane_infos:
            if self.availability(plane) and plane['ID'] != myself['ID'] and\
                    plane['ID'] in self.lasttime_has_layout and self.lasttime_has_layout[plane['ID']] == select_plane['ID']:
                plane_theta = self.XY2theta(select_plane['X']-plane['X'], select_plane['Y']-plane['Y'])
                d_theta = self.pi_bound(plane_theta - my_theta)
                if d_theta >= 0:
                    right_num += 1
                else:
                    left_num += 1


        if my_distance <= turn_distance:
            goal_R = goal_R2
        else:
            goal_R = goal_R1

        turn_theta = math.pi - math.acos(goal_R / my_distance)
        dest_theta = my_theta
        if right_num == 0:
            dest_theta = self.pi_bound(dest_theta - turn_theta)
        if left_num == 0:
            dest_theta = self.pi_bound(dest_theta + turn_theta)

        dest_x = select_plane['X'] + goal_R * math.sin(dest_theta)
        dest_y = select_plane['Y'] + goal_R * math.cos(dest_theta)

        return {'X': dest_x, 'Y': dest_y, 'Z': data['area_max_alt']-500}


    def rect_bound(self, dest, R):
        if dest['X'] < -R:
            dest['X'] = -R
        if dest['X'] > R:
            dest['X'] = R
        if dest['Y'] < -R:
            dest['Y'] = -R
        if dest['Y'] > R:
            dest['Y'] = R
        return dest


    def layout_plane_select(self, myself):
        if self.is_revenge_time() == False:
            return {'plane': self.enemy_manned_info[0], 'mission': 'attack'}
        else:
            min_distance = 450000
            min_enemy = {}
            for enemy in self.enemy_allplane_infos:
                if self.availability(enemy):
                    distance = TSVector3.distance(enemy, self.my_manned_info[0])
                    if distance < min_distance:
                        min_distance = distance
                        min_enemy = enemy
            return {'plane': min_enemy, 'mission': 'attack'}


    def get_KeppDistance_move_XY(self, myself, danger, distance):
        bound = 145000
        danger_to_myself_theta = self.XY2theta(myself['X'] - danger['X'], myself['Y'] - danger['Y'])
        dest_x = danger['X'] + distance * math.sin(danger_to_myself_theta)
        dest_y = danger['Y'] + distance * math.cos(danger_to_myself_theta)

        if dest_x > bound:
            dest_x = bound
        if dest_x < -bound:
            dest_x = -bound
        if dest_y > bound:
            dest_y = bound
        if dest_y < -bound:
            dest_y = -bound

        return {'X': dest_x, 'Y': dest_y}



    def attack_select(self, myself):

        attack_list = self.enemy_uvas_infos + self.enemy_manned_info

        return attack_list


    def is_revenge_time(self):
        return self.my_manned_info[0]['CurTime'] >= 20 * 60 - 120 or \
               len(self.my_allplane_infos) < len(self.enemy_allplane_infos) or\
               self.my_manned_info[0]['CurTime'] >= 20 * 60 -300 and self.my_score <= self.enemy_score


    def availability(self, item):
        return item['ID'] != 0 and item['Availability'] > 0.0001


    def surround_dict(self, item, enemy_list):
        allplane = [item]
        for enemy in enemy_list:
            double_flag = 0
            for plane in allplane:
                if plane['X'] == enemy['X'] and plane['Y'] == enemy['Y']:
                    double_flag = 1
            if double_flag == 0:
                allplane.append(enemy)
        s_dict = {}
        for plane0 in allplane:
            s_dict[plane0['ID']] = {}
            for plane1 in allplane:
                s_dict[plane0['ID']][plane1['ID']] = {'distance': -1, 'dx': 0, 'dy': 0}
        for plane0 in allplane:
            for plane1 in allplane:
                if plane0['ID'] != plane1['ID']:
                    aside_flag = 1
                    sign_flag = 0
                    for plane2 in allplane:
                        if plane2['ID'] != plane0['ID'] and plane2['ID'] != plane1['ID']:
                            side = self.pi_bound(
                                self.XY2theta(plane2['X'] - plane0['X'], plane2['Y'] - plane0['Y']) - self.XY2theta(
                                    plane1['X'] - plane0['X'], plane1['Y'] - plane0['Y']))
                            if side > 0 and sign_flag == -1 or side < 0 and sign_flag == 1:
                                aside_flag = 0
                                break
                            if side > 0:
                                sign_flag = 1
                            elif side < 0:
                                sign_flag = -1
                    if aside_flag == 1:
                        s_dict[plane0['ID']][plane1['ID']]['distance'] = abs(((plane1['X'] - plane0['X']) * (
                                item['Y'] - plane0['Y']) - (plane1['Y'] - plane0['Y']) * (item['X'] - plane0[
                            'X'])) / ((plane1['X'] - plane0['X']) ** 2 + (plane1['Y'] - plane0['Y']) ** 2) ** 0.5)
                        theta = self.XY2theta(plane1['X'] - plane0['X'], plane1['Y'] - plane0['Y'])
                        if self.pi_bound(
                                self.XY2theta(item['X'] - plane0['X'], item['Y'] - plane0['Y']) - self.XY2theta(
                                    plane1['X'] - plane0['X'], plane1['Y'] - plane0['Y'])) > 0:
                            theta = self.pi_bound(theta - math.pi / 2)
                        else:
                            theta = self.pi_bound(theta + math.pi / 2)
                        s_dict[plane0['ID']][plane1['ID']]['dx'] = math.sin(theta)
                        s_dict[plane0['ID']][plane1['ID']]['dy'] = math.cos(theta)
        return s_dict


    def get_num_locking_my_manner(self):
        num = 0
        for missile in self.enemy_missile_infos:
            if self.availability(missile) and missile['EngageTargetID'] == self.my_manned_info[0]['ID']:
                num += 1
        return num


    def get_danger_speed(self, item):
        danger_move_v = 1000
        if item['Type'] == 3 and item['CurTime'] - self.missile_init_time[item['ID']] > 30:
            danger_move_v = item['Speed']
        return danger_move_v


    def get_tangent_shot_time(self, item0, item1):
        d = TSVector3.distance(item0, item1)
        shot_t = d / self.get_danger_speed(item0)
        enemy_turn_t = 0
        if item0['Type'] != 3:
            enemy_turn_t = self.get_fire_turn_time(item0, item1, 1)
        return shot_t + enemy_turn_t


    def get_danger_shot_time(self, item0, item0_v, item1, item1_v, theta):
        d = TSVector3.distance(item0, item1)
        alpha = self.pi_bound(theta - self.XY2theta(item1['X'] - item0['X'], item1['Y'] - item0['Y']))
        shot_t = (item0_v + item1_v * math.cos(alpha)) * d / (item0_v ** 2 - item1_v ** 2)
        enemy_turn_t = 0
        if item0['Type'] != 3:
            enemy_turn_t = self.get_fire_turn_time(item0, item1, 1)
        my_turn_t = self.get_escape_turn_time(item1, theta, 0)
        return shot_t + enemy_turn_t - my_turn_t


    def get_escape_turn_time(self, myself, theta, flag):
        my_turn_theta = abs(self.pi_bound(theta - myself['Heading']))
        data = self.get_move_data(myself)
        if flag == 1:
            my_turn_t = my_turn_theta / (data['move_max_g'] * 9.8) * data['move_min_speed']
        else:
            my_turn_t = my_turn_theta / (data['move_max_g'] * 9.8) * myself['Speed']
        return my_turn_t


    def get_fire_turn_time(self, myself, enemy, flag):
        my_turn_theta = abs(
            self.pi_bound(self.XY2theta(enemy['X'] - myself['X'], enemy['Y'] - myself['Y']) - myself['Heading']))
        look_theta = math.pi / 3
        if myself['Type'] == 2:
            look_theta /= 2
        if my_turn_theta > look_theta:
            my_turn_theta -= look_theta
        else:
            my_turn_theta = 0
        data = self.get_move_data(myself)
        if flag == 1:
            my_turn_t = my_turn_theta / (data['move_max_g'] * 9.8) * data['move_min_speed']
        else:
            my_turn_t = my_turn_theta / (data['move_max_g'] * 9.8) * myself['Speed']
        return my_turn_t


    def dest_bound(self, plane, dest_x, dest_y, max_bound, dest_distance):
        R = (dest_x * dest_x + dest_y * dest_y) ** 0.5
        dest = {'X': dest_x, 'Y': dest_y}
        if R > max_bound:
            r = (plane['X'] ** 2 + plane['Y'] ** 2) ** 0.5
            if r > (150000 + max_bound) / 2:  
                d_theta = self.pi_bound(plane['Heading'] - self.XY2theta(plane['X'], plane['Y']))
                toward = 1
                if d_theta < 0:
                    toward = -1
                dest_theta = self.pi_bound(self.XY2theta(plane['X'], plane['Y']) + toward * math.pi / 18)
                old_dest_theta = self.XY2theta(dest['X'] - plane['X'], dest['Y'] - plane['Y'])
                old_d_theta = self.pi_bound(old_dest_theta - self.XY2theta(plane['X'], plane['Y']))
                old_toward = 1
                if old_d_theta < 0:
                    old_toward = -1
                if old_toward == toward and R < r:
                    dest_theta = old_dest_theta
                dest['X'] = max_bound * math.sin(dest_theta)
                dest['Y'] = max_bound * math.cos(dest_theta)
                return dest

            plane_theta = self.XY2theta(plane['X'], plane['Y'])

            dx = dest_x - plane['X']
            dy = dest_y - plane['Y']
            turn_theta = 2 * math.atan(dest_distance / 2 / max_bound)  
            if dx * plane['Y'] - dy * plane['X'] < 0:  
                turn_theta = -turn_theta
            dest_theta = self.pi_bound(plane_theta + turn_theta)

            dest['X'] = max_bound * math.sin(dest_theta)
            dest['Y'] = max_bound * math.cos(dest_theta)
        return dest


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


    def pi_bound(self, theta):
        while theta > math.pi:
            theta -= 2*math.pi
        while theta <= -math.pi:
            theta += 2*math.pi
        return theta


    def is_out_of_map(self, x, y):
        if abs(x) > 145000 or abs(y) > 145000:
            return True
        return False


    def is_in_center(self, plane):
        distance_to_center = (plane['X']**2 + plane["Y"]**2 + (plane["Alt"] - 9000)**2)**0.5
        if distance_to_center <= 50000 and plane["Alt"] >= 2000 and plane['Alt'] <= 16000:
            return True
        return False

    def is_looking(self, item0, item1, item0_toward_theta):
        data = self.get_move_data(item0)
        looking_theta = math.pi/3
        if item0['Type'] == 2:
            looking_theta /= 2
        item_distance = TSVector3.distance(item0, item1)
        item_theta = self.XY2theta(item1['X'] - item0['X'], item1['Y'] - item0['Y'])
        if item_distance < data['launch_range'] and \
                abs(self.pi_bound(item_theta - item0_toward_theta)) < looking_theta:
            return True
        return False

    def is_attacking(self, item0, item1):
        for missile in self.enemy_missile_infos+self.my_missile_infos:
            if self.availability(missile) and missile['LauncherID'] == item0['ID'] and missile['EngageTargetID'] == item1['ID']:
                return True
        return False

    def get_attack_area(self, myself, enemy):
        data0 = self.get_move_data(myself)
        data1 = self.get_move_data(enemy)

        my_turn_t = self.get_fire_turn_time(myself, enemy, 0)
        enemy_escape_theta = self.XY2theta(enemy['X']-myself['X'], enemy['Y']-myself['Y'])
        enemy_turn_t = self.get_escape_turn_time(enemy, enemy_escape_theta, 1)

        enemy_escape_v = data1['move_max_speed']
        my_missile_end_v = 550

        max_area = 55000
        if my_turn_t < enemy_turn_t:
            max_area += my_missile_end_v * (enemy_turn_t - my_turn_t)
        else:
            max_area += enemy_escape_v * (enemy_turn_t - my_turn_t)

        if max_area > data0['launch_range']:
            max_area = data0['launch_range']

        return max_area


    def get_plane_by_id(self, planeID) -> {}:

        for plane in self.my_allplane_infos + self.enemy_allplane_infos:
            if plane['ID']== planeID:
                return plane
        return None

    def get_move_data(self, plane) -> {}:
        data = {}
        if plane['Type'] == 1:
            data['move_min_speed'] = 150
            data['move_max_speed'] = 400
            data['move_max_acc'] = 1
            data['move_max_g'] = 6
            data['area_max_alt'] = 14000
            data['attack_range'] = 1
            data['launch_range'] = 80000
        else:
            data['move_min_speed'] = 100
            data['move_max_speed'] = 300
            data['move_max_acc'] = 2
            data['move_max_g'] = 12
            data['area_max_alt'] = 10000
            data['attack_range'] = 1
            data['launch_range'] = 60000
        return data

    def get_missile_data(self):
        missile_path_distance = [0, 542.2063609607588, 1182.480494005152, 1920.822152703217, 2757.231069395307,
                                 3691.7069649739587, 4690.380668163883, 5690.392781982494, 6690.403824273029,
                                 7690.413884262445, 8690.423067290565, 9690.431489479675, 10690.439270851693,
                                 11690.446527674249, 12690.453364428302, 13690.459866663177, 14690.46609559915,
                                 15690.472085141579, 16690.47784150794, 17690.483345493267, 18690.48855694438,
                                 19690.49342097367, 20690.49792117785, 21690.50208666525, 22690.505926345282,
                                 23690.50945277755, 24690.512678259773, 25690.51561484032, 26690.518274292805,
                                 27690.52066813158, 28685.055583652498, 29669.786997780793, 30644.90639985083,
                                 31610.599720751903, 32567.047546045003, 33514.42531895895, 34452.90353383846,
                                 35382.647920574476, 36303.81962054315, 37216.57535445111, 38121.06758263005,
                                 39017.44465807968, 39905.850972742825, 40786.42709728389, 41659.30991479319,
                                 42524.63274866314, 43382.525484983555, 44233.11468971713, 45076.52372092886,
                                 45912.872836297705, 46742.279296177825, 47564.85746240277, 48380.71889305735,
                                 49189.97243339209, 49992.72430310625, 50789.07818013033, 51579.13528109239,
                                 52362.99443863512, 53140.752175714144, 53912.5027770251, 54678.33835768422,
                                 55438.348929307605, 56192.62246358879, 56941.24495348287, 57684.3004721222,
                                 58421.87122955169, 59154.03762738674, 59880.878311491746, 60602.47022271564,
                                 61318.88864586478, 62030.20725687547, 62736.49816835767, 63437.83197351848,
                                 64134.2777885513, 64825.90329358807, 65512.774772186196, 66194.95714952274,
                                 66872.51402926158, 67545.50772917981, 68213.99931561596, 68878.04863675927,
                                 69537.71435483627, 70193.05397725917, 70844.12388674797, 71490.97937045498,
                                 72133.67464820115, 72772.26289976719, 73406.79629133553, 74037.32600108047,
                                 74663.90224398884, 75286.57429587955, 75905.39051667579, 76520.39837297822,
                                 77131.6444599387, 77739.17452247204, 78343.03347581085, 78943.26542547542,
                                 79539.9136866049, 80133.02080275358, 80722.62856410189, 81308.77802514988,
                                 81891.50952189989, 82470.86268850697, 83046.87647350151, 83619.58915550118,
                                 84189.03835849119, 84755.2610666802, 85318.29363892187, 85878.17182274409,
                                 86434.9307679779, 86988.60504001667, 87539.22863270907, 88086.83498088206,
                                 88631.45697255083, 89173.12696077435, 89711.87677520081, 90247.73773330044,
                                 90780.74065129971, 91310.91585482856, 91838.29318928291]

        missile_v = [498.06649999999985, 596.1329999999998, 694.1995000000001, 792.2660000000003,
                     890.3325000000003, 988.3990000000001, 999.9999999999999, 1000.0000000000001,
                     1000.0000000000001, 1000.0000000000001, 1000.0, 1000.0000000000001, 1000.0000000000001,
                     999.9999999999999, 999.9999999999999, 1000.0, 999.9999999999999, 1000.0, 1000.0, 1000.0,
                     1000.0, 1000.0, 999.9999999999999, 1000.0000000000001, 999.9999999999999, 999.9999999999999,
                     999.9999999999999, 1000.0000000000001, 1000.0000000000001, 998.9999999999998,
                     989.1089644855925, 979.4119640272487, 969.9033426243097, 960.5776620373183, 951.4296914059851,
                     942.4543974556145, 933.646935253426, 925.0026394790765, 916.5170161763232, 908.185734955161,
                     900.0046216159873, 891.9696511693809, 884.0769412269634, 876.3227457405142, 868.7034490681223,
                     861.2155603476166, 853.8557081588581, 846.6206354577488, 839.5071947659442, 832.5123436013392,
                     825.6331401353875, 818.8667390642127, 812.2103876813401, 805.6614221406508, 799.217263898891,
                     792.875416327757, 786.6334614861918, 780.4890570441256, 774.4399333494351, 768.4838906303997,
                     762.6187963264058, 756.8425825400944, 751.1532436045508, 745.5488337595231, 740.0274649310105,
                     734.587304608892, 729.2265738175863, 723.9435451750174, 718.7365410354355, 713.6039317118918,
                     708.5441337744205, 703.5556084201772, 698.636859912022, 693.7864340822124, 689.002916898051,
                     684.2849330865361, 679.6311448151782, 675.040250426344, 670.5109832225997, 666.0421103006762,
                     661.6324314317978, 657.280777986242, 652.9860119000974, 648.7470246823115, 644.5627364601941,
                     640.4320950616608, 636.3540751325714, 632.3276772876087, 628.3519272932188, 624.425875281212,
                     620.5485949916877, 616.719183044016, 612.936758234671, 609.2004608607668, 605.5094520682105,
                     601.8629132234253, 598.2600453076608, 594.7000683329479, 591.1822207788008, 587.7057590488099,
                     584.2699569463135, 580.8741051683674, 577.5175108172766, 574.1994969289791, 570.9194020176066,
                     567.6765796355792, 564.4703979486226, 561.3002393251141, 558.1654999392041, 555.0655893871682,
                     551.9999303164889, 548.9679580671668, 545.9691203248012, 543.0028767849832, 540.0686988285847,
                     537.1660692075228, 534.2944817406116, 531.4534410191269, 528.6424621217202, 525.8610703383404]
        return {'path_distance': missile_path_distance, 'v': missile_v}