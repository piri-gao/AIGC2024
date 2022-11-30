""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/8/24 10:00
"""
import uuid
import numpy as np
from agent.agent import Agent
from agent.jixianAI.utils import get_angle_deg, get_dis

class AttackPolicy():

    def __attack_policy_init__(self,):
        # 每轮结束后初始化这个函数
        self.cooperative_combat_dic = {}
        self.threat_rank_list = []

        for my_jet in self.my_jets:
            if not hasattr(my_jet, "attack_status"):
                my_jet.attack_status = "default"

        for enemy_jet in self.enemy_jets:
            enemy_jet.under_multiple_attack = 0
            enemy_jet.under_solo_attack = 0

    def attack_policy(self):
        if self.cooperative_combat_dic is None:
            self.__attack_policy_init__()

        # 更新导弹的分配，为每个攻击任务预留出导弹
        self.update_missile_allocation()
        # 更新每架敌机受到的最大交叉火力角度和可以进行这个角度打击的我方飞机
        self.update_enemy_cross_fire_angle()
        # 对敌人进行威胁排序
        self.update_threat_rank()

    def update_missile_allocation(self):
        for my_jet in self.my_jets:
            my_jet.hold_missile = 0
            if my_jet.attack_status == "lead_waiting":
                # 如果长机正在等待副攻的导弹就位，那么为长机预留一发导弹
                my_jet.hold_missile += 1
                coop = self.cooperative_combat_dic[my_jet.lead_task_id]
                if coop["third_missile_needed"]:
                    my_jet.hold_missile += 1

            for ms_todo in my_jet.ms_to_launch:
                if ms_todo["launch_status"] is None:
                    my_jet.hold_missile += 1
                elif ms_todo["launch_status"] == "given_cmd":
                    # 筛选出刚发射的导弹
                    launch_finished = [
                        ms for ms in self.missile
                        if ms.EngageTargetID == ms_todo["missile_target_id"] and
                        ms.LauncherID == my_jet.ID and
                        ms.flying_time == 0
                    ]
                    if len(launch_finished) == 0:
                        # 虽然是given_cmd阶段，但导弹发射没能执行成功
                        my_jet.hold_missile += 1
                    if len(launch_finished) == 2:
                        print("异常情况，需要检查")

        for my_jet in self.my_jets:
            if "有人机" in my_jet.Name:
                my_jet.unavail_missile = 1
                my_jet.avail_missile = my_jet.LeftWeapon - my_jet.unavail_missile - my_jet.hold_missile
            elif "无人机" in my_jet.Name:
                my_jet.unavail_missile = 0
                my_jet.avail_missile = my_jet.LeftWeapon - my_jet.unavail_missile - my_jet.hold_missile

    def update_enemy_cross_fire_angle(self):
        for enemy_jet in self.enemy_jets:
            max_cross_fire_angle = 0
            cross_fire_team = None
            for p1 in self.my_jets:
                for p2 in self.my_jets:
                    if p2 is p1 :
                        continue
                    delta_deg = get_angle_deg(p1, enemy_jet, p2)
                    if delta_deg <= max_cross_fire_angle:
                        continue
                    max_cross_fire_angle = delta_deg
                    cross_fire_team = {"p1": p1, "p2": p2}
            enemy_jet.max_cross_fire_angle = max_cross_fire_angle
            enemy_jet.cross_fire_team = cross_fire_team

    def update_threat_rank(self):

        # 对于当前无需打击的目标，将威胁值降低到极小值
        t0 = 999999

        # 距离
        t1 = 1.0 / 1e3
        nearest_dis = np.array([min([get_dis(enemy_jet, my_jet) for my_jet in self.my_jets]) for enemy_jet in self.enemy_jets])
        threat_array = -nearest_dis * t1

        t2 = 30
        t3 = 40
        t4 = -50

        my_avail_jets = [my_jet for my_jet in self.my_jets if my_jet.LeftWeapon >= 0]
        for i, enemy_jet in enumerate(self.enemy_jets):
            # 是否已经被攻击
            # 如果敌机是无人机并且在被协同打击，则无需攻击
            if enemy_jet.under_multiple_attack > 0 and enemy_jet.Tyep == 2:
                threat_array[i] = threat_array[i] - t0
            if enemy_jet.Type == 1:
                # 如果目标是有人机默认提高50的威胁值
                threat_array[i] -= t4
                if enemy_jet.under_multiple_attack == 1:
                    threat_array[i] = threat_array[i] - t2
                elif enemy_jet.under_multiple_attack >= 2:
                    threat_array[i] = threat_array[i] - t0

            # 如果敌机在被单机打击任务打击，则无需攻击
            if enemy_jet.under_solo_attack:
                threat_array[i] = threat_array[i] - t0

            # 是否被交叉火力覆盖
            sorted_my_jets = sorted(my_avail_jets, key=lambda jet: get_dis(jet, enemy_jet))
            dis = np.array([get_dis(enemy_jet, my_jet) for my_jet in sorted_my_jets])
            for j in range(len(dis)-1):
                if np.abs(dis[j+1] - dis[j]) < 5e3:
                    delta_deg = get_angle_deg(p1=sorted_my_jets[j], p2=sorted_my_jets[j+1], p_center=enemy_jet)
                    if delta_deg >= 45:
                        threat_array[i] += t3
                        break

            # 如果是无弹目标，则无需攻击
            if enemy_jet.EnemyLeftWeapon <= 0:
                threat_array[i] -= t0

            enemy_jet.threat_level = threat_array[i]

        sorted_threat = sorted(self.enemy_jets, key=lambda enemy_jet: -enemy_jet.threat_level)
        filtered_sorted_threat = list(filter(lambda enemy_jet: enemy_jet.threat_level > -99999, sorted_threat))

        self.threat_rank_list = filtered_sorted_threat

    def assign_coop_attack_tasks(self):
        for enemy_jet in self.threat_rank_list:
            # 遍历每一架敌机，初筛可以作为长机攻击它的我方战机
            # 有可用的弹药；不是长机；必须是无人机；在设定的attack_dis中
            my_canlead_jets = [
                jet for jet in self.my_jets
                if (jet.avail_missile > 0) and
                ("lead" not in jet.attack_status) and
                jet.Type == 2 and
                get_dis(jet, enemy_jet) < jet.attack_dis
            ]
            if len(my_canlead_jets) == 0:
                print("没有适合作为长机的飞机")
                return
            sorted_canlead_jets = sorted(my_canlead_jets, key=lambda p: get_dis(enemy_jet, p))

            tmp_cooperative_uid = []
            # 选择最合适的长机
            for lead_jet in sorted_canlead_jets:
                # 先赋予主机的状态，防止等下选择副攻的时候选到待评估的长机
                lead_jet.attack_status = "lead_waiting"
                coop_result, coop_uuid = self.choose_wingman(lead_jet, enemy_jet)
                if coop_result == "coop_success":
                    tmp_cooperative_uid.append(coop_uuid)
                lead_jet.attack_status = "default"




    def choose_wingman(self, lead_jet, enemy_jet):
        lead_dis = get_dis(lead_jet, enemy_jet)

        can_winman_jets = []
        # 筛选可以作为僚机的我方战机
        for my_jet in self.my_jets:
            # 必须要有弹药；和目标的距离必须大于长机和目标的距离
            tmp_dis = get_dis(my_jet, enemy_jet)
            if my_jet.avail_missile > 0 and tmp_dis > lead_dis:
                # 不是长机并且在合适攻击的距离内
                condition1 = ("lead" not in my_jet.attack_status) and (tmp_dis < my_jet.RaderDis)
                # 是长机但敌机正好在雷达探测范围内
                condition2 = ("lead" in my_jet.attack_status) and (enemy_jet in my_jet.in_rader)
                if condition1 or condition2:
                    can_winman_jets.append(enemy_jet)

        if len(can_winman_jets) <= 0:
            return None

        def evaluate_winman(lead_jet, wingman_jet, enemy_jet):
            # 评估原则，距离越近越好，角度越大越好
            dis = get_dis(enemy_jet, wingman_jet)
            delta_deg = get_angle_deg(p1=wingman_jet, p_center=enemy_jet, p2=lead_jet)
            assert delta_deg <= 180
            constant = 10e3 / 45
            return dis - (delta_deg - 45) * constant

        can_winman_jets_rank = sorted(can_winman_jets, key=lambda p: evaluate_winman(lead_jet, p, enemy_jet))

        winman_jet = can_winman_jets_rank[0]
        cross_fire_angle = get_angle_deg(p1=winman_jet, p_center=enemy_jet, p2=lead_jet)

        # 只有在敌机是有人机并且我方弹药充足（大于1枚）的情况下才允许一架敌机被三枚导弹打击
        third_missile_needed = True if (enemy_jet.Type == 1 and winman_jet.avail_missile >= 2) else False

        coop_uuid = uuid.uuid1().hex
        self.cooperative_combat_dic[coop_uuid] = {
            "target" : enemy_jet,
            "lead_jet": lead_jet,
            "wingman_jet": winman_jet,
            "coop_uuid": coop_uuid,
            "lead_missile": None,
            "wingman_missile": None,
            "third_missile_needed": third_missile_needed,
            "cross_fire_angle": cross_fire_angle,
            "is_valid": False
        }

        return "coop_success",coop_uuid




