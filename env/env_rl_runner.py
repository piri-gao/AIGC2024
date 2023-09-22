import os
import math
import random
from time import sleep
from config import config_rl as config
from env.xsim_env import XSimEnv


LEADER_SCORE_WEIGHT = 16  # 有人机得分权重
UAV_SCORE_WEIGHT = 4     # 无人机得分权重
MISSILE_SCORE_WEIGHT = 0.2  # 导弹得分权重
CENTER_AREA_RADIO = 50000

red_area_score = 0
blue_area_score = 0
count = 0

DONE_REASON_MSG = {
    "0": "未达到终止条件!",
    "1": "超时!",
    "2": "红方剩余兵力不足!",
    "3": "蓝方剩余兵力不足!",
    "4": "双方剩余兵力都不足!",
}


class EnvRunner(XSimEnv):
    def __init__(self, address):
        print("初始化 EnvRunner")
        XSimEnv.__init__(self, config['time_ratio'], address)
        self.agents = {}
        self.__init_agents()
        self.start_time = 7*60
        self.launch_missile = []
        self.last_red_entities = []
        self.last_blue_entities = []
        self.damage_entities = []

    def get_env_info(self):
        env_info = {
            "n_agents":10,
            "obs_shape":196,
            "state_shape":610,
            "n_actions":12+3,
            "episode_limit":20*60 - self.start_time
        }
        return env_info

    def __init_agents(self):
        self.red_cls = config["agents"]['red']
        self.blue_cls = config["agents"]['blue']
        red_agent = self.red_cls('red', {"side": 'red'})
        self.rl_side = 'red'
        self.tink_side = 'blue'
        self.agents["red"] = red_agent
        # blue_index = random.randint(0, len(self.blue_cls)-1)
        blue_index = -1
        self.agents["blue"] = self.blue_cls[blue_index]('blue', {"side": 'blue'})

    def _reset(self):
        # 智能体重置
        # sleep(5)
        self.__init_agents()
        # 环境重置
        self.reset()
        self.rew_obs = self.step([])
        while self.rew_obs is None:
            sleep(0.2)
            self.rew_obs = self.step([])
        self.obs = None
        self.state = None
        self.avail_actions = None
        self.cur_time = 0
    
    def get_obs(self):
        if self.obs is None:
            sleep(0.2)
            self.rew_obs = self.step([])
            while self.rew_obs is None:
                sleep(0.2)
                self.rew_obs = self.step([])
            self.cur_time = self.rew_obs["sim_time"]
            while self.cur_time<=3:
                self.rew_obs = self.step([])
                self.cur_time = self.rew_obs["sim_time"]
            my_obs,global_obs,reward,done,actions_mask,info=self.agents[self.rl_side].step(self.cur_time, self.rew_obs[self.rl_side])
            self.obs = my_obs
            self.state = global_obs
            self.avail_actions = actions_mask
        return self.obs

    def get_state(self):
        return self.state
    
    def get_avail_actions(self):
        return self.avail_actions
    
    def action2order(self, actions):
        cmd_list = []
        self.agents[self.rl_side].action2order(actions, cmd_list)
        return cmd_list

    def _step(self, a_n):
        actions = []
        fixed_act,cmd_list = self.agents[self.rl_side].update_decision(self.rew_obs[self.rl_side])
        if self.cur_time>self.start_time:
            cmd_list = self.action2order(a_n)
        actions.extend(cmd_list)
        cmd_list = self.agents[self.tink_side].step(self.cur_time, self.rew_obs[self.tink_side])
        actions.extend(cmd_list)
        self.rew_obs = self.step(actions)
        while self.rew_obs is None:
                sleep(0.1)
                self.rew_obs = self.step(actions)
        # print(self.rew_obs)
        self.cur_time = self.rew_obs["sim_time"]
        my_obs,global_obs,reward,done,actions_mask,info=self.agents[self.rl_side].step(self.cur_time, self.rew_obs[self.rl_side])
        self.obs = my_obs
        self.state = global_obs
        self.avail_actions = actions_mask
        win = self.get_done()  # 推演结束(分出胜负或达到最大时长)
        if win[0]:  # 对战结束后环境重置
            done[done==0] = 1
            print("到达终止条件！！！！")
            if win[1] > win[2]:
                print("该局 ：  红方胜",  " 红方分数", win[1],"   蓝方分数", win[2],)
                info['battle_won'] = True
            else:
                print("该局 ：  蓝方胜",  " 红方分数", win[1],"   蓝方分数", win[2],)
                info['battle_won'] = False
        # print(self.cur_time,done)
        return reward, done, info, fixed_act

    def get_done(self):
        global red_area_score
        global blue_area_score
        done = [0, 0, 0]  # 终止标识， 红方战胜利， 蓝方胜利
        # 时间超时，终止
        if self.cur_time >= 20 * 60 - 1:
            done[0] = 1
            done = self._print_score(done, "1", red_area_score, blue_area_score)

            # 重置区域得分
            red_area_score = 0
            blue_area_score = 0
            return done

        red_obs_units = self.rew_obs["red"]["platforminfos"]
        for red_obs_unit in red_obs_units:
            if red_obs_unit["Type"] == 1:
                # 判断红方有人机是否在中心区域
                distance_to_center = math.sqrt(red_obs_unit["X"] * red_obs_unit["X"] +
                                               red_obs_unit["Y"] * red_obs_unit["Y"] +
                                               (red_obs_unit["Alt"] - 9000) * (red_obs_unit["Alt"] - 9000))
                if distance_to_center <= CENTER_AREA_RADIO and red_obs_unit["Alt"] >= 2000 and red_obs_unit['Alt'] <= 16000:
                    red_area_score = red_area_score + 1

        blue_obs_units = self.rew_obs["blue"]["platforminfos"]
        for blue_obs_unit in blue_obs_units:
            if blue_obs_unit["Type"] == 1:
                # 判断蓝方有人机是否在中心区域
                distance_to_center = math.sqrt(blue_obs_unit["X"] * blue_obs_unit["X"] +
                                               blue_obs_unit["Y"] * blue_obs_unit["Y"] +
                                               (blue_obs_unit["Alt"] - 9000) * (blue_obs_unit["Alt"] - 9000))
                if distance_to_center <= CENTER_AREA_RADIO and \
                        blue_obs_unit["Alt"] >= 2000 and \
                        blue_obs_unit['Alt'] <= 16000:
                    blue_area_score = blue_area_score + 1

        red_man_num = 0
        red_uav_num = 0
        blue_man_num = 0
        blue_uav_num = 0

        for red_obs_unit in red_obs_units:
            if red_obs_unit["Type"] == 1:
                red_man_num += 1
            if red_obs_unit["Type"] == 2:
                red_uav_num += 1

        for blue_obs_unit in blue_obs_units:
            if blue_obs_unit["Type"] == 1:
                blue_man_num += 1
            if blue_obs_unit["Type"] == 2:
                blue_uav_num += 1

        red_done = False
        if red_man_num == 0 or\
            (red_man_num == 1 and red_uav_num < 4) or\
            (red_man_num == 2 and red_uav_num == 0):
            red_done = True

        blue_done = False
        if blue_man_num == 0 or \
            (blue_man_num == 2 and blue_uav_num == 0) or\
            (blue_man_num == 1 and blue_uav_num < 4):
            blue_done = True

        if red_done and not blue_done:
            done[0] = 1
            done[2] = 1
            red_area_score = 0
            blue_area_score = 0
            # red_round_score, blue_round_score = self._cal_score(obs)
            done = self._print_score(done, "2")
            return done

        if blue_done and not red_done:
            done[0] = 1
            done[1] = 1
            red_area_score = 0
            blue_area_score = 0
            # red_round_score, blue_round_score = self._cal_score(obs)
            done = self._print_score(done, "3")
            return done

        if red_done and blue_done:
            done[0] = 1
            red_area_score = 0
            blue_area_score = 0
            done = self._print_score(done, "4")
            return done
        return done

    @staticmethod
    def _cal_score(obs):

        # 计算剩余兵力
        red_leader = 0
        red_uav = 0
        red_missile = 0
        blue_leader = 0
        blue_uav = 0
        blue_missile = 0
        for unit in obs["red"]["platforminfos"]:
            red_missile += unit["LeftWeapon"]
            if unit["Type"] == 1:
                red_leader += 1
            else:
                red_uav += 1
        for unit in obs["blue"]["platforminfos"]:
            blue_missile += unit["LeftWeapon"]
            if unit["Type"] == 1:
                blue_leader += 1
            else:
                blue_uav += 1

        # 计算剩余兵力与剩余导弹数的权重和

        red_round_score = red_leader * LEADER_SCORE_WEIGHT + \
                          red_uav * UAV_SCORE_WEIGHT + \
                          red_missile * MISSILE_SCORE_WEIGHT
        blue_round_score = blue_leader * LEADER_SCORE_WEIGHT + \
                           blue_uav * UAV_SCORE_WEIGHT + \
                           blue_missile * MISSILE_SCORE_WEIGHT

        return red_round_score, blue_round_score


    def print_logs(self, num):
        global count
        if self.rew_obs is None:
            return 
        filename = "logs/" + str(self.red_cls).split("'")[1] + "_VS_" + str(self.blue_cls).split("'")[1] + "_" + str(
            num) + ".txt"
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if num != count:
            if not os.path.isdir("logs"):
                os.mkdir("logs")
            if os.path.isfile(filename):
                os.remove(filename)
            count = num
            with open(filename, "w") as file:
                file.write("红方:"+ str(self.red_cls).split("'")[1] +"\n")
                file.write("蓝方:" + str(self.blue_cls).split("'")[1] + "\n")
                file.close()

            self.last_red_entities = self.rew_obs["red"]["platforminfos"]
            self.last_blue_entities = self.rew_obs["blue"]["platforminfos"]
            self.launch_missile = []
            self.damage_entities = []

        with open(filename, "a") as fileobject:
            cur_red_entity_ids = []
            for cur_red_entity in self.rew_obs["red"]["platforminfos"]:
                cur_red_entity_ids.append(cur_red_entity["ID"])
            for last_red_entity in self.last_red_entities:
                # 如果上一步长的实体不在当前步长实体内,说明该实体毁伤
                # 记录毁伤实体信息,用于匹配导弹信息
                if last_red_entity["ID"] not in cur_red_entity_ids:
                    self.damage_entities.append(last_red_entity)
                    entity_file = "[" + str(self.cur_time) + "][" + str(last_red_entity["Identification"]) + "]:[" + last_red_entity[
                        "Name"] + "]战损"
                    fileobject.writelines(entity_file + "\n")
            self.last_red_entities = self.rew_obs["red"]["platforminfos"]

            cur_blue_entity_ids = []
            for cur_blue_entity in self.rew_obs["blue"]["platforminfos"]:
                cur_blue_entity_ids.append(cur_blue_entity["ID"])
            for last_blue_entity in self.last_blue_entities:
                # 如果上一步长的实体不在当前步长实体内,说明该实体毁伤
                # 记录毁伤实体信息,用于匹配导弹信息
                if last_blue_entity["ID"] not in cur_blue_entity_ids:
                    self.damage_entities.append(last_blue_entity)
                    entity_file = "[" + str(self.cur_time) + "][" + str(last_blue_entity["Identification"]) + "]:[" + \
                                  last_blue_entity["Name"] + "]战损"
                    fileobject.writelines(entity_file + "\n")
            self.last_blue_entities = self.rew_obs["blue"]["platforminfos"]

            all_entities = self.last_red_entities + self.last_blue_entities + self.damage_entities
            for missile in self.rew_obs["red"]["missileinfos"]:
                if missile["ID"] not in self.launch_missile:
                    EngageTargetName = None
                    self.launch_missile.append(missile["ID"])
                    # for entity in all_entities:
                    #     if entity["ID"] == missile["LauncherID"]:
                    #         LauncherName = entity["Name"]
                    #         break
                    for entity in all_entities:
                        if entity["ID"] == missile["EngageTargetID"]:
                            EngageTargetName = entity["Name"]
                            break
                    # rocket_file = "[" + str(cur_time - 1) + "][" + str(
                    #     missile["Identification"]) + "]:[" + LauncherName + "]发射一枚导弹打击[" + EngageTargetName + "]"
                    rocket_file = "[" + str(self.cur_time - 1) + "][" + str(
                        missile["Identification"]) + "]:" + "发射一枚导弹打击[" + EngageTargetName + "]"
                    fileobject.writelines(rocket_file + "\n")


    def _print_score(self, done, done_reason_code = "0", red_area_score = 0, blue_area_score = 0):
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        filename = "logs/" + str(self.red_cls).split("'")[1] + "_VS_" + str(self.blue_cls).split("'")[1] + "_" + str(
            count) + ".txt"
        with open(filename, "a") as fileobject:
            # 输出结束原因
            fileobject.writelines("到达终止条件: " + DONE_REASON_MSG[done_reason_code] + "\n")
            red_round_score, blue_round_score = self._cal_score(self.rew_obs)
            # 胜负判断
            if done[1] > done[2]:
                fileobject.writelines("红方获胜!" + "\n")
            elif done[1] < done[2]:
                fileobject.writelines("蓝方获胜!" + "\n")
            else:
                done[1] = red_round_score
                done[2] = blue_round_score
                if red_round_score > blue_round_score:
                    fileobject.writelines("红方获胜!" + "\n")
                if red_round_score < blue_round_score:
                    fileobject.writelines("蓝方获胜!" + "\n")
                if red_round_score == blue_round_score:
                    done[2] = blue_area_score
                    done[1] = red_area_score
                    if red_area_score or blue_area_score:
                        if red_area_score > blue_area_score:
                            fileobject.writelines("双方得分相同.但红方占据中心位置的时间较长!" + "\n")
                            fileobject.writelines("红方获胜!" + "\n")
                        elif red_area_score < blue_area_score:
                            fileobject.writelines("双方得分相同.但蓝方占据中心位置的时间较长!" + "\n")
                            fileobject.writelines("蓝方获胜!" + "\n")
                        else:
                            fileobject.writelines("双方得分相同.且双方占据中心位置的时间一样长!" + "\n")
                            fileobject.writelines("平局!" + "\n")
                    else:
                        fileobject.writelines("平局!" + "\n")

            fileobject.writelines("红方得分:" + str(red_round_score) + "\n")
            fileobject.writelines("蓝方得分:" + str(blue_round_score) + "\n")

            return done
