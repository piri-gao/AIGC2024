"""
完全版AI 防守反击策略
"""
from typing import List
from agent.agent import Agent
from  env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
import  copy
import  random
import math

"""
选手需要重写继承自基类Agent中的step(self,sim_time, obs_red, **kwargs)去实现自己的策略逻辑，注意，这个方法每个步长都会被调用
"""

class LiuBY_Infinite_Agent(Agent):
    """
         自定义的Demo智能体
     @Examples:
         添加使用示例
         >>> 填写使用说明
         ··· 填写简单代码示例
     """
    def __init__(self, name, config):
        """
        初始化信息
        :param name:阵营名称
        :param config:阵营配置信息
        """
        super(LiuBY_Infinite_Agent, self).__init__(name, config["side"])
        self._init() #调用用以定义一些下文所需变量

    def _init(self):
        """对下文中使用到的变量进行定义和初始化"""
        self.my_uvas_infos = []         #该变量用以保存己方所有无人机的态势信息
        self.my_manned_info = []        #该变量用以保存己方有人机的态势信息
        self.my_allplane_infos = []     #该变量用以保存己方所有飞机的态势信息
        self.my_missile_infos = []     
        self.enemy_uvas_infos = []      #该变量用以保存敌方所有无人机的态势信息
        self.enemy_manned_info = []     #该变量用以保存敌方有人机的态势信息
        self.enemy_allplane_infos = []  #该变量用以保存敌方所有飞机的态势信息
        self.enemy_missile_infos = []   #该变量用以保存敌方导弹的态势信息
        self.enemy_leftweapon = {}      
        self.regiments = []            

        self.attack_handle_enemy = {}   

        self.missile_path_length = {} 
        self.missile_last_pisition = {}
        self.missile_init_time = {} 
        self.attacked_flag = {}  
        self.curtime_has_attack = {} 

        self.safe_flag = False
        self.attack_dict = {}

        self.escape_toward = {}  
        self.escape_last_dangerID = {} 

        self.missile_to_enemy_manned = 0
        self.missile_to_enemy_uvas = 0

        self.my_score = 0
        self.enemy_score = 0

        self.my_allplane_maps = {}

        self.missile_path_distance = self.get_missile_data()['path_distance']
        self.missile_v = self.get_missile_data()['v']
        self.terminal_guidance_distance = 20000

    def reset(self, **kwargs):
        self.my_uvas_infos.clear()  # 该变量用以保存己方所有无人机的态势信息
        self.my_manned_info.clear()  # 该变量用以保存己方有人机的态势信息
        self.my_allplane_infos.clear()  # 该变量用以保存己方所有飞机的态势信息
        self.my_missile_infos.clear()   #该变量用以保存我方导弹的态势信息
        self.enemy_uvas_infos.clear()  # 该变量用以保存敌方所有无人机的态势信息
        self.enemy_manned_info.clear()  # 该变量用以保存敌方有人机的态势信息
        self.enemy_allplane_infos.clear()  # 该变量用以保存敌方所有飞机的态势信息
        self.enemy_missile_infos.clear()  # 该变量用以保存敌方导弹的态势信息
        self.enemy_leftweapon.clear()  # 该变量用以保存敌方飞机的导弹量

        self.attack_handle_enemy.clear()

        self.missile_path_length.clear()  
        self.missile_last_pisition.clear() 
        self.missile_init_time.clear()   
        self.attacked_flag.clear()        
        self.curtime_has_attack.clear()

        self.safe_flag = False
        self.attack_dict.clear()

        self.escape_toward.clear()  
        self.escape_last_dangerID.clear() 

        self.missile_to_enemy_manned = 0
        self.missile_to_enemy_uvas = 0

        self.my_score = 0
        self.enemy_score = 0

        self.my_allplane_maps.clear()

        pass

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        
        cmd_list = []   # 此变量为 保存所有的决策完毕任务指令列表

        self.process_decision(sim_time, obs_side, cmd_list) # 调用决策函数进行决策判断

        return cmd_list # 返回决策完毕的任务指令列表

    def process_decision(self, sim_time, obs_side, cmd_list):
        
        self.process_observation(sim_time, obs_side)  # 获取态势信息,对态势信息进行处理

        if sim_time == 1:  # 当作战时间为1s时,初始化实体位置,注意,初始化位置的指令只能在前三秒内才会被执行
            self.init_pos(cmd_list) # 将实体放置到合适的初始化位置上

        if sim_time >= 2:  # 当作战时间大于10s时,开始进行任务控制,并保存任务指令;
            self.attack_dict = self.attack_select() 
            self.process_move(sim_time, cmd_list)   

    def process_observation(self, sim_time, obs_side):
        """
        初始化飞机态势信息
        :param obs_red: 当前方所有的态势信息信息，包含了所有的当前方所需信息以及探测到的敌方信息
        """
        my_entity_infos = obs_side['platforminfos'] # 拿到己方阵营有人机、无人机在内的所有飞机信息
        if len(my_entity_infos) < 1:
            return
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

        if len(my_manned_info) < 1:       #  判断己方有人机是否被摧毁
            return

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

        my_allplane_maps = {}
        for input_entity in my_allplane_infos:
            my_allplane_maps[int(input_entity['ID'])] = input_entity
            self.my_allplane_maps[int(input_entity['ID'])] = input_entity

        missile_infos = obs_side['missileinfos']  # 拿到空间中已发射且尚未爆炸的导弹信息

        my_missile_infos = []       #  用以保存己方已发射且尚未爆炸的导弹信息
        enemy_missile_infos = []    #  用以保存敌方已发射且尚未爆炸的导弹信息

        for missile_info in missile_infos:
            if (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001): # 判断导弹是否可用 导弹的ID即为导弹的唯一编号 导弹的Availability为导弹当前生命值
                if missile_info['LauncherID'] in self.my_allplane_maps:  # 判断导弹是否为己方导弹 导弹的LauncherID即为导弹的发射者
                    missile_info["Z"] = missile_info["Alt"]     # 导弹的 Alt 即为导弹的当前高度
                    my_missile_infos.append(missile_info)       # 保存己方已发射且尚未爆炸的导弹信息
                else:
                    missile_info["Z"] = missile_info["Alt"]     # 导弹的 Alt 即为导弹的当前高度
                    enemy_missile_infos.append(missile_info)    # 保存敌方已发射且尚未爆炸的导弹信息

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

        self.my_uvas_infos = my_uvas_infos              # 保存当前己方无人机信息
        self.my_manned_info = my_manned_info            # 保存当前己方有人机信息
        self.my_allplane_infos = my_allplane_infos            # 保存当前己方所有飞机信息
        self.my_missile_infos = my_missile_infos     # 保存己方已发射且尚未爆炸的导弹信息
        self.enemy_uvas_infos = enemy_uvas_infos        # 保存当前敌方无人机信息
        self.enemy_manned_info = enemy_manned_info      # 保存当前敌方有人机信息
        self.enemy_allplane_infos = enemy_allplane_infos      # 保存当前敌方所有飞机信息
        self.enemy_missile_infos = enemy_missile_infos  # 保存敌方已发射且尚未爆炸的导弹信息

        
        for enemy_plane_ID in self.enemy_leftweapon: 
            enemy_plane = self.get_object_by_id(enemy_plane_ID)
            if enemy_plane is not None:
                enemy_plane['LeftWeapon'] = self.enemy_leftweapon[enemy_plane_ID]

        for plane_ID in self.attacked_flag: 
            plane = self.get_object_by_id(plane_ID)
            if plane is not None:
                plane['LockedMissileID'] = self.attacked_flag[plane_ID]

        for missile_ID in self.missile_init_time:
            missile = self.get_object_by_id(missile_ID)
            if missile is not None:
                missile['InitTime'] = self.missile_init_time[missile_ID]

        
        regiment_distance = 60000
        self.regiments = []
        for plane in self.enemy_allplane_infos: 
            new_regiment_flag = True
            for regiment in self.regiments:
                far_flag = True
                for other_plane in regiment:
                    if TSVector3.distance(plane, other_plane) < regiment_distance:
                        far_flag = False
                        break
                if far_flag:
                    continue
                else:
                    new_regiment_flag = False
                    regiment.append(plane)
            if new_regiment_flag:
                self.regiments.append([plane])
            else:
                new_regiment = []
                for regiment in self.regiments[-1: : -1]:
                    if plane in regiment:
                        self.regiments.remove(regiment)
                        regiment.remove(plane)
                        new_regiment += regiment
                new_regiment.append(plane)
                self.regiments.append(new_regiment)


        for i in range(len(self.regiments)): 
            if self.enemy_manned_info[0] in self.regiments[i]:
                t = self.regiments[i]
                self.regiments[i] = self.regiments[-1]
                self.regiments[-1] = t
                break
        for i in range(len(self.regiments)-2): 
            for j in range(len(self.regiments)-i-2):
                if len(self.regiments[j]) > len(self.regiments[j+1]):
                    t = self.regiments[j]
                    self.regiments[j] = self.regiments[j+1]
                    self.regiments[j+1] = t
        
        for enemy_plane in self.enemy_allplane_infos: 
            enemy_plane['RegimentID'] = -1
            for i in range(len(self.regiments)):
                if enemy_plane in self.regiments[i]:
                    enemy_plane['RegimentID'] = i

        for my_plane in self.my_allplane_infos: 
            my_plane['RegimentID'] = -1
        for regiment_ID in range(len(self.regiments)):
            i = 0
            if self.my_manned_info[0]['RegimentID'] == -1: 
                self.my_manned_info[0]['RegimentID'] = 0
                i += 1
            regiment = self.regiments[regiment_ID]
            while i <= len(regiment): 
                min_distance = 450000
                min_uvas = {}
                for my_uvas in self.my_uvas_infos:
                    if my_uvas['RegimentID'] == -1 and my_uvas['LeftWeapon'] > 0:
                        for enemy_plane in regiment:
                            if TSVector3.distance(enemy_plane, my_uvas) < min_distance:
                                min_distance = TSVector3.distance(enemy_plane, my_uvas)
                                min_uvas = my_uvas
                
                if len(min_uvas) == 0:
                    break
                min_uvas['RegimentID'] = regiment_ID
                i += 1
        for my_plane in self.my_uvas_infos:
            if my_plane['RegimentID'] == -1 and my_plane['LeftWeapon'] > 0: 
                my_plane['RegimentID'] = 0
        while True: 
            rest_no_weapon_uvas_flag = False
            for my_plane in self.my_uvas_infos:
                if my_plane['RegimentID'] == -1 and my_plane['LeftWeapon'] == 0:
                    rest_no_weapon_uvas_flag = True
                    break
            if not rest_no_weapon_uvas_flag:
                break
            for regiment_ID in range(len(self.regiments)):
                regiment = self.regiments[regiment_ID]
                min_distance = 450000
                min_uvas = {}
                for my_uvas in self.my_uvas_infos:
                    if my_uvas['RegimentID'] == -1 and my_uvas['LeftWeapon'] == 0:  
                        for enemy_plane in regiment:
                            if TSVector3.distance(enemy_plane, my_uvas) < min_distance:
                                min_distance = TSVector3.distance(enemy_plane, my_uvas)
                                min_uvas = my_uvas
                if len(min_uvas) == 0:
                    break
                min_uvas['RegimentID'] = regiment_ID

        for regiment in self.regiments:
            E_x = 0
            E_y = 0
            weapon_plane_num = 0
            for enemy_plane in regiment:
                if enemy_plane['LeftWeapon'] > 0:
                    weapon_plane_num += 1
                    E_x += enemy_plane['X']
                    E_y += enemy_plane['Y']
            if weapon_plane_num == 0: 
                continue
            E_x /= weapon_plane_num
            E_y /= weapon_plane_num
            for enemy_plane in regiment:
                enemy_plane['EX'] = E_x
                enemy_plane['EY'] = E_y

        for i in range(len(self.regiments)):
            regiment = []
            for my_plane in self.my_allplane_infos:
                if my_plane['RegimentID'] == i:
                    regiment.append(my_plane)
            if len(regiment) == 0:
                continue
            E_x = 0
            E_y = 0
            weapon_uvas_num = 0
            for my_plane in regiment:
                if my_plane['Type'] == 2 and my_plane['LeftWeapon'] > 0:
                    weapon_uvas_num += 1
                    E_x += my_plane['X']
                    E_y += my_plane['Y']
            if weapon_uvas_num == 0: 
                continue
            E_x /= weapon_uvas_num
            E_y /= weapon_uvas_num
            for my_plane in regiment:
                my_plane['EX'] = E_x
                my_plane['EY'] = E_y


        for i in range(len(self.regiments)): 
            regiment = self.regiments[i]
            if len(regiment) <= 1:
                for my_plane in self.my_allplane_infos:
                    if my_plane['RegimentID'] == i:
                        my_plane['LayoutID'] = regiment[0]['ID']
            else:
                E_x = 0 
                E_y = 0
                for enemy_plane in regiment:
                    E_x += enemy_plane['X']
                    E_y += enemy_plane['Y']
                E_x /= len(regiment)
                E_y /= len(regiment)
                main_vector_theta = 0
                max_V = -1
                for item in range(180):
                    theta = item / 360 * math.pi
                    enemy_V = 0
                    for enemy_plane in regiment:
                        d_enemy_x = enemy_plane['X'] - E_x
                        d_enemy_y = enemy_plane['Y'] - E_y
                        d_enemy_r = d_enemy_x * math.sin(theta) + d_enemy_y * math.cos(theta)
                        enemy_V += d_enemy_r * d_enemy_r
                    if enemy_V > max_V:
                        main_vector_theta = theta
                        max_V = enemy_V
                my_x = 0
                my_y = 0
                my_regiment_plane = {}
                for my_plane in self.my_allplane_infos:
                    if my_plane['RegimentID'] == i:
                        my_regiment_plane = my_plane
                        break
                if len(my_regiment_plane) == 0 or 'EX' not in my_regiment_plane:
                    my_x = self.my_manned_info[0]['X']
                    my_y = self.my_manned_info[0]['Y']
                else:
                    my_x = my_regiment_plane['EX']
                    my_y = my_regiment_plane['EY']
                d_my_x = my_x - E_x
                d_my_y = my_y - E_y
                d_my_r = d_my_x * math.sin(main_vector_theta) + d_my_y * math.cos(main_vector_theta)
                toward = 1
                if d_my_r < 0:
                    toward = -1
                max_r = 0
                layout_plane = {}
                for enemy_plane in regiment:
                    d_enemy_x = enemy_plane['X'] - E_x
                    d_enemy_y = enemy_plane['Y'] - E_y
                    d_enemy_r = d_enemy_x * math.sin(main_vector_theta) + d_enemy_y * math.cos(main_vector_theta)
                    if d_enemy_r * toward >= max_r:
                        max_r = d_enemy_r * toward
                        layout_plane = enemy_plane
                for my_plane in self.my_allplane_infos:
                    if my_plane['RegimentID'] == i:
                        my_plane['LayoutID'] = layout_plane['ID']

        self.manned_danger_flag = False
        for enemy_plane in self.enemy_allplane_infos:
            if enemy_plane['LeftWeapon'] > 0 and TSVector3.distance(enemy_plane, self.my_manned_info[0]) < 40000:
                self.manned_danger_flag = True
                break
        if self.manned_danger_flag:
            min_distance = 450000
            layout_uvas = {}
            for enemy_uvas in self.enemy_uvas_infos:
                enemy_distance = TSVector3.distance(enemy_uvas, self.my_manned_info[0])
                if enemy_uvas['LeftWeapon'] > 0 and enemy_distance < min_distance:
                    min_distance = enemy_distance
                    layout_uvas = enemy_uvas
            if len(layout_uvas) > 0:
                for my_plane in self.my_allplane_infos:
                    if my_plane['LeftWeapon'] > 2 - my_plane['Type'] and my_plane['RegimentID'] == 0:
                        my_plane['LayoutID'] = layout_uvas['ID']


        self.safe_flag = self.is_safe()

    def init_pos(self, cmd_list):
        leader_original_pos = {}    # 用以初始化当前方的位置
        leader_heading = 0
        if self.name == "red":
            leader_original_pos = {"X": -125000, "Y": 0, "Z": 9000}
            leader_heading = 90
        else :
            leader_original_pos = {"X": 125000, "Y": 0, "Z": 9000}
            leader_heading = 270

        interval_distance = 5000   # 间隔 5000米排列
        for leader in self.my_manned_info: # 为己方有人机设置初始位置
            cmd_list.append(CmdEnv.make_entityinitinfo(leader['ID'], leader_original_pos['X'], leader_original_pos['Y'], leader_original_pos['Z'],400, leader_heading))

        #己方无人机在有人机的y轴上分别以9500的间距进行部署
        sub_index = 0  # 编号 用以在有人机左右位置一次排序位置点
        for sub in self.my_uvas_infos: # 为己方每个无人机设置初始位置
            sub_pos = copy.deepcopy(leader_original_pos)  # 深拷贝有人机的位置点
            if sub_index & 1 == 0: # 将当前编号放在有人机的一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] + interval_distance, sub_pos['Z'],300 * 0.6, leader_heading))
            else:                   # 将当前编号放在有人机的另一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] - interval_distance, sub_pos['Z'],300 * 0.6, leader_heading))
                #interval_distance *= 2 # 编号翻倍
                interval_distance *= 2
            sub_index += 1 # 编号自增


    def attack_select(self):
        can_attack_dict = {}
        for enemy_plane in self.enemy_allplane_infos:
            can_attack_dict[enemy_plane['ID']] = []

        curtime_attack_myplane = {}
        for my_plane in self.my_allplane_infos:
            curtime_attack_myplane[my_plane['ID']] = False

        # 有人机被贴脸时

        danger_enemy_uvas_num = 0
        for enemy_uvas in self.enemy_uvas_infos:
            if enemy_uvas['LeftWeapon'] > 0:
                danger_enemy_uvas_num += 1
        if danger_enemy_uvas_num > 0:
            my_leader = self.my_manned_info[0]
            for plane in self.my_allplane_infos:
                for enemy_plane in self.enemy_allplane_infos:
                    if TSVector3.distance(enemy_plane, my_leader) < 8000 and len(
                            self.attacked_flag[enemy_plane['ID']]) + self.curtime_has_attack[enemy_plane['ID']] <= 1:
                        if plane['Type'] == 1 and plane['LeftWeapon'] > 1 and not self.is_attacking(plane, enemy_plane) and self.is_looking(plane, enemy_plane, plane['Heading']):
                            can_attack_dict[enemy_plane['ID']].append(plane['ID'])
                            curtime_attack_myplane[plane['ID']] = True
                            self.curtime_has_attack[enemy_plane['ID']] += 1
                        elif plane['Type'] == 2 and plane['LeftWeapon'] > 0 and not self.is_in_cold(plane, enemy_plane):
                            min_flag = True
                            for my_uvas in self.my_uvas_infos:
                                if my_uvas['LeftWeapon'] > 0 and not self.is_in_cold(my_uvas, enemy_plane) and \
                                        TSVector3.distance(my_uvas, enemy_plane) < TSVector3.distance(plane, enemy_plane):
                                    min_flag = False
                            if min_flag: #立刻救援
                                can_attack_dict[enemy_plane['ID']].append(plane['ID'])
                                curtime_attack_myplane[plane['ID']] = True
                                self.curtime_has_attack[enemy_plane['ID']] += 1



        attacked_list_1 = []
        for enemy_plane in self.enemy_uvas_infos + self.enemy_manned_info:
            enemy_looking_flag = False
            for my_plane in self.my_allplane_infos:
                if self.is_looking(enemy_plane, my_plane, enemy_plane['Heading']) and self.is_guidanced(my_plane):
                    enemy_looking_flag = True
                    break
            if enemy_looking_flag and len(enemy_plane['LockedMissileID']) + self.curtime_has_attack[enemy_plane['ID']] == 0:
                attacked_list_1.append(enemy_plane)

        attacked_list_2 = []
        for enemy_plane in self.enemy_uvas_infos + self.enemy_manned_info:
            if enemy_plane not in attacked_list_1:
                attacked_list_2.append(enemy_plane)

        for enemy_plane in attacked_list_1 + attacked_list_2:
            if enemy_plane in attacked_list_1:
                best_plane = {}
                min_distance = 450000
                for my_plane in self.my_allplane_infos:
                    data = self.get_move_data(my_plane)
                    if not curtime_attack_myplane[my_plane['ID']] and not self.is_attacking(my_plane, enemy_plane) and \
                            (my_plane['LeftWeapon'] > 0 and TSVector3.distance(enemy_plane, my_plane) < data['launch_range'] and my_plane['Type'] == 2 or \
                            my_plane['LeftWeapon'] > 1 and TSVector3.distance(enemy_plane, my_plane) < data['launch_range'] and my_plane['Type'] == 1 and \
                            self.is_looking(my_plane, enemy_plane, my_plane['Heading'])):
                        distance = TSVector3.distance(my_plane, enemy_plane)
                        if distance < min_distance:
                            min_distance = distance
                            best_plane = my_plane
                if len(best_plane) != 0:
                    can_attack_dict[enemy_plane['ID']].append(best_plane['ID'])
                    curtime_attack_myplane[best_plane['ID']] = True
                    self.curtime_has_attack[enemy_plane['ID']] += 1
            elif enemy_plane['Type'] == 2:
                rest = 2-self.curtime_has_attack[enemy_plane['ID']]-len(enemy_plane['LockedMissileID'])
                for i in range(rest):
                    best_plane = {}
                    min_distance = 450000
                    max_leftweapon = -1
                    for my_plane in self.my_allplane_infos:
                        data = self.get_move_data(my_plane)
                        if (my_plane['LeftWeapon'] > 1 and my_plane['Type'] == 1 or my_plane['LeftWeapon'] > 0 and my_plane['Type'] == 2) and\
                                not self.is_attacking(my_plane, enemy_plane) and not curtime_attack_myplane[my_plane['ID']] and \
                                TSVector3.distance(enemy_plane, my_plane) < self.get_attack_area(my_plane, enemy_plane):
                            distance = TSVector3.distance(my_plane, enemy_plane)
                            if my_plane['LeftWeapon'] > max_leftweapon or my_plane['LeftWeapon'] == max_leftweapon and distance < min_distance:
                                max_leftweapon = my_plane['LeftWeapon']
                                min_distance = distance
                                best_plane = my_plane
                    if len(best_plane) != 0:
                        can_attack_dict[enemy_plane['ID']].append(best_plane['ID'])
                        curtime_attack_myplane[best_plane['ID']] = True
                        self.curtime_has_attack[enemy_plane['ID']] += 1

        if self.safe_flag:  # 合击猎杀敌方有人机
            # 有载弹飞机在敌方15000米内 且当前自己是最远的可开火（任意夹角 > pi/3）飞机时 开火（非冷却）
            # 有导弹击中时间小于我的击中时间 且该导弹与我方夹角 > pi/3时 开火（非冷却）
            enemy_leader = self.enemy_manned_info[0]

            max_distance = -1 #计算导火索飞机
            trigger_plane = {}
            for plane in self.my_allplane_infos:
                if not self.is_in_cold(plane, enemy_leader) and plane['LeftWeapon'] > 2 - plane['Type'] and \
                        (plane['Type'] == 1 and (80000 > TSVector3.distance(plane, enemy_leader) > 76000 or
                                                 TSVector3.distance(plane, enemy_leader) < 20000) or \
                         plane['Type'] == 2 and TSVector3.distance(plane, enemy_leader) < 60000):
                    theta_flag = False
                    for combine_plane in self.my_allplane_infos:
                        if combine_plane['ID'] != plane['ID'] and \
                                combine_plane['LeftWeapon'] > 2 - combine_plane['Type'] and \
                                TSVector3.distance(combine_plane, enemy_leader) < 15000:
                            combine_plane_theta = self.XY2theta(combine_plane['X']-enemy_leader['X'],
                                                                combine_plane['Y']-enemy_leader['Y'])
                            plane_theta = self.XY2theta(plane['X']-enemy_leader['X'],
                                                        plane['Y']-enemy_leader['Y'])
                            d_theta = abs(self.pi_bound(combine_plane_theta-plane_theta))
                            if d_theta > math.pi/3:
                                theta_flag = True
                                break
                    if theta_flag and max_distance < TSVector3.distance(plane, enemy_leader):
                        max_distance = TSVector3.distance(plane, enemy_leader)
                        trigger_plane = plane

            for my_plane in self.my_allplane_infos:
                if not self.is_in_cold(my_plane, enemy_leader) and my_plane['LeftWeapon'] > 2 - my_plane['Type'] and \
                        (my_plane['Type'] == 1 and 80000 > TSVector3.distance(my_plane, enemy_leader) or \
                         my_plane['Type'] == 2 and TSVector3.distance(my_plane, enemy_leader) < 60000):
                    fire_flag = False
                    fire_type = 0
                    if len(trigger_plane) != 0 and trigger_plane['ID'] == my_plane['ID']: # trigger开火
                        fire_flag = True
                        fire_type = 1
                    if not fire_flag and TSVector3.distance(my_plane, enemy_leader) < 20000: # combine开火
                        for trigger_missile in self.my_missile_infos:
                            trigger_missile_theta = self.XY2theta(trigger_missile['X'] - enemy_leader['X'],
                                                                  trigger_missile['Y'] - enemy_leader['Y'])
                            my_plane_theta = self.XY2theta(my_plane['X'] - enemy_leader['X'],
                                                           my_plane['Y'] - enemy_leader['Y'])
                            d_theta = abs(self.pi_bound(trigger_missile_theta - my_plane_theta))
                            if d_theta > math.pi/3 and abs(self.get_tangent_shot_time(my_plane, enemy_leader) - self.get_tangent_shot_time(trigger_missile, enemy_leader)) < 10:
                                fire_flag = True
                                fire_type = 2
                                break
                    if fire_flag:
                        can_attack_dict[enemy_leader['ID']].append(my_plane['ID'])
                        curtime_attack_myplane[my_plane['ID']] = True
                        self.curtime_has_attack[enemy_leader['ID']] += 1






        return can_attack_dict

    def enemy_has_accompany(self, enemy, myself):
        if TSVector3.distance(myself, enemy) > self.get_attack_area(myself, enemy):
            return False
        myself_theta = self.XY2theta(myself['X'] - enemy['X'], myself['Y'] - enemy['Y'])
        for plane in self.my_uvas_infos:
            if plane['ID'] != myself['ID'] and plane['LeftWeapon'] > 0 and TSVector3.distance(plane, enemy) < 7000:
                plane_theta = self.XY2theta(plane['X'] - enemy['X'], plane['Y'] - enemy['Y'])
                myself_plane_theta = abs(self.pi_bound(myself_theta-plane_theta))
                if myself_plane_theta >= math.pi/2:
                    return True
        return False


    def is_safe(self):
        if len(self.enemy_uvas_infos) == 0 and self.enemy_manned_info[0]['LeftWeapon'] <= 1 and len(self.enemy_missile_infos) == 0:
            return True
        return False


    def process_move(self, sim_time, cmd_list):

        for plane in self.my_allplane_infos:
            self.single_process_move(sim_time, cmd_list, plane)


    def single_process_move(self, sim_time, cmd_list, plane):

        if len(self.my_manned_info) > 0:

            for enemy_ID in self.attack_dict.keys():
                enemy_plane = self.get_object_by_id(enemy_ID)
                if plane['ID'] in self.attack_dict[enemy_ID] and \
                        (self.is_looking(plane, enemy_plane, plane['Heading']) or
                         enemy_plane['Type'] == 1 and plane['Type'] == 2):
                    cmd_list.append(CmdEnv.make_attackparam(plane['ID'], enemy_ID, 1))
                    return

            dest_distance = 10000  
            max_bound = 145000 
            enemy_manned = self.enemy_manned_info[0]
            danger_tangent_distance = 30000 
            danger_uvas_tangent_distance = 50000
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
                    max_attack_range = self.missile_path_distance[120] - self.missile_path_distance[int(sim_time-missile['InitTime'])] + 200
                    if distance > max_attack_range + \
                            data['move_max_speed'] * (120-sim_time+missile['InitTime'] - \
                            self.get_escape_turn_time(plane, self.XY2theta(missile['X']-plane['X'], missile['Y']-plane['Y']), 0)): 
                        continue
                    missile_time = self.get_tangent_shot_time(missile, plane) 
                    danger_list.append(missile)
                    if missile_time < lock_min_time:
                        lock_min_distacne = distance
                        lock_min_time = missile_time
                        lock_min_danger = missile

            for enemy_plane in self.enemy_allplane_infos: 
                if self.availability(enemy_plane) and (enemy_plane['LeftWeapon'] > 0 and plane['Type'] == 1 or
                                                       enemy_plane['LeftWeapon'] > 2-enemy_plane['Type'] and plane['Type'] == 2): 
                    distance = TSVector3.distance(plane, enemy_plane)
                    if distance < danger_tangent_distance and plane['Type'] == 1 or \
                            distance < danger_uvas_tangent_distance and plane['Type'] == 2: 
                        enemy_time = self.get_tangent_shot_time(enemy_plane, plane) 
                        danger_list.append(enemy_plane)
                        if enemy_time < lock_min_time:
                            lock_min_distacne = distance
                            lock_min_time = enemy_time
                            lock_min_danger = enemy_plane

            if len(danger_list) > 0 and not (self.safe_flag and plane['Type'] == 2): 
                tangent_flag = False
                dx = 0
                dy = 0
                if plane['Type'] == 1:
                    if self.can_escape(lock_min_danger, plane,
                            self.XY2theta(plane['X'] -lock_min_danger['X'], plane['Y']-lock_min_danger['Y'])) is False and \
                            lock_min_danger['Type'] != 1:
                        tangent_flag = True
                        if lock_min_danger['Type'] == 3 and lock_min_time + lock_min_danger['CurTime'] - lock_min_danger['InitTime'] > 84: 
                            dx_ = lock_min_danger['X'] - plane['X']
                            dy_ = lock_min_danger['Y'] - plane['Y']
                            r = (dx_ * dx_ + dy_ * dy_) ** 0.5
                            if r < 0.0001:
                                r = 0.0001
                            dx = dx_ * dest_distance / r 
                            dy = dy_ * dest_distance / r
                            self.escape_toward[plane['ID']] = 0
                        else:

                            dx_ = plane['X'] - lock_min_danger['X']
                            dy_ = plane['Y'] - lock_min_danger['Y']
                            r = (dx_ * dx_ + dy_ * dy_) ** 0.5
                            if r < 0.0001:
                                r = 0.0001
                            dx = -dy_ * dest_distance / r  
                            dy = dx_ * dest_distance / r

                            if self.escape_last_dangerID[plane['ID']] != lock_min_danger['ID']:  
                                self.escape_last_dangerID[plane['ID']] = lock_min_danger['ID']
                                if lock_min_time < 15:
                                    self.escape_toward[plane['ID']] = 0

                            if self.escape_toward[plane['ID']] == 1:  
                                dx = -dx
                                dy = -dy
                            elif self.escape_toward[plane['ID']] == 0:
                                left_danger = 0
                                right_danger = 0
                                for danger in danger_list:
                                    if danger['ID'] != lock_min_danger['ID']:
                                        if self.pi_bound(self.XY2theta(lock_min_danger['X'] - plane['X'], lock_min_danger['Y'] - plane['Y']) -
                                                         self.XY2theta(danger['X'] - plane['X'], danger['Y'] - plane['Y'])) >= 0:
                                            left_danger += 1
                                        else:
                                            right_danger += 1
                                if dx * plane['X'] + dy * plane['Y'] >= 0 and (plane['X'] ** 2 + plane['Y'] ** 2) ** 0.5 >= max_bound - dest_distance \
                                        or (dx * math.sin(plane['Heading']) + dy * math.cos(plane['Heading']) < 0 and TSVector3.distance(lock_min_danger, plane) < 15000 \
                                        or left_danger <= right_danger and TSVector3.distance(lock_min_danger, plane) >= 15000) and \
                                        (plane['X'] ** 2 + plane['Y'] ** 2) ** 0.5 < max_bound - dest_distance:
                                    dx = -dx
                                    dy = -dy
                                    self.escape_toward[plane['ID']] = 1
                                else:
                                    self.escape_toward[plane['ID']] = -1
                else:
                    if self.can_escape(lock_min_danger, plane,
                            self.XY2theta(plane['X']-lock_min_danger['X'], plane['Y']-lock_min_danger['Y'])) is False:
                        tangent_flag = True
                        dx_ = lock_min_danger['X'] - plane['X']
                        dy_ = lock_min_danger['Y'] - plane['Y']
                        r = (dx_ * dx_ + dy_ * dy_) ** 0.5
                        if r < 0.0001:
                            r = 0.0001
                        dx = dx_ * dest_distance / r  
                        dy = dy_ * dest_distance / r
                if tangent_flag:
                    dest_x = plane['X'] + dx
                    dest_y = plane['Y'] + dy
                    dest = self.dest_bound(plane, dest_x, dest_y, max_bound, dest_distance)
                    if plane['Type'] == 2:
                        dest = self.rect_dest_bound(dest_x, dest_y, max_bound)
                    leader_fire_route_list = [{"X": dest['X'], "Y": dest['Y'],
                                               "Z": data['area_max_alt'] - 500}, ]
                    cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                data['move_max_speed'], data['move_max_acc'],
                                                                data["move_max_g"]));
                    return






            locking_list = [] 
            locking_min_enemy = {}
            locking_min_time = 2000
            for my_missile in self.my_missile_infos:
                if self.availability(my_missile) and my_missile['LauncherID'] == plane['ID']:
                    enemy_plane = self.get_object_by_id(my_missile['EngageTargetID'])
                    if enemy_plane != None and self.availability(enemy_plane) and\
                            TSVector3.distance(my_missile, enemy_plane) >= free_distance:
                        locking_list.append(enemy_plane)
                        data_enemy = self.get_move_data(enemy_plane)
                        locking_time = self.get_danger_shot_time(my_missile, enemy_plane, enemy_plane['Heading'])
                        if locking_time < locking_min_time:
                            locking_min_enemy = enemy_plane
                            locking_min_time = locking_time
            locking_min_distacne = 450000
            if len(locking_list) == 0:
                for my_missile in self.my_missile_infos:
                    if self.availability(my_missile):
                        enemy_plane = self.get_object_by_id(my_missile['EngageTargetID'])
                        if enemy_plane != None and self.availability(enemy_plane) and enemy_plane['RegimentID'] == plane['RegimentID'] and \
                                TSVector3.distance(my_missile, enemy_plane) >= free_distance:
                            locking_distance = TSVector3.distance(enemy_plane, plane)
                            if locking_distance > data['launch_range']:
                                continue
                            locking_list.append(enemy_plane)
                            data_enemy = self.get_move_data(enemy_plane)
                            locking_time = self.get_danger_shot_time(my_missile, enemy_plane, enemy_plane['Heading'])
                            if locking_distance < locking_min_distacne or locking_distance == locking_min_distacne and locking_time < locking_min_time:
                                locking_min_enemy = enemy_plane
                                locking_min_time = locking_time
                                locking_min_distacne = locking_distance


            if len(locking_list) > 0:

                self.escape_toward[plane['ID']] = 0 

                only_track_flag = 0 

                myplane_enemy_theta = self.XY2theta(locking_min_enemy['X']-plane['X'], locking_min_enemy['Y']-plane['Y'])
                track_theta = self.pi_bound(plane['Heading'] - myplane_enemy_theta)
                enemy_theta = self.pi_bound(locking_min_enemy['Heading'] - myplane_enemy_theta)

                myplane_enemy_distance = TSVector3.distance(plane, locking_min_enemy)
                if myplane_enemy_distance < 0.0001:
                    cmd_list.append(CmdEnv.make_followparam(plane['ID'], locking_min_enemy['ID'],
                                                            data['move_max_speed'],
                                                            data['move_max_acc'], data['move_max_g']))
                    return

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
                return



            if len(danger_list) > 0 and not (self.is_safe() and plane['Type'] == 2): 
                self.escape_toward[plane['ID']] = 0  
                dx_ = plane['X'] - lock_min_danger['X']
                dy_ = plane['Y'] - lock_min_danger['Y']
                r = (dx_ * dx_ + dy_ * dy_) ** 0.5
                if r < 0.0001:
                    r = 0.0001
                dx = dx_ * dest_distance / r
                dy = dy_ * dest_distance / r
                dest_x = plane['X'] + dx
                dest_y = plane['Y'] + dy
                dest = self.dest_bound(plane, dest_x, dest_y, max_bound, dest_distance)
                if plane['Type'] == 2:
                    dest = self.rect_dest_bound(dest_x, dest_y, max_bound)
                leader_fire_route_list = [{"X": dest['X'], "Y": dest['Y'],
                                           "Z": data['area_max_alt'] - 500}, ]
                if math.sin(plane['Heading']) * (dest_x - plane['X']) + math.cos(plane['Heading']) * (dest_y - plane['Y']) > 0:
                    cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                data['move_max_speed'], data['move_max_acc'],
                                                                data["move_max_g"]));
                else:
                    cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                data['move_min_speed'], data['move_max_acc'],
                                                                data["move_max_g"]));
                return

            for enemy_ID in self.attack_dict.keys():
                enemy_plane = self.get_object_by_id(enemy_ID)
                if plane['ID'] in self.attack_dict[enemy_ID]:
                    cmd_list.append(CmdEnv.make_followparam(plane['ID'], enemy_ID, data['move_max_speed'], data['move_max_acc'],
                                                            data['move_max_g']))
                    return



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
            if surround_flag == 1 and not (self.safe_flag and plane['Type'] == 2):

                self.escape_toward[plane['ID']] = 0 

                dest_x = 0
                dest_y = 0
                if plane['Type'] == 1 or plane['Type'] == 2:
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
                return

            else: 
                self.escape_toward[plane['ID']] = 0 
                if len(self.enemy_manned_info) > 0:
                    dest = self.layout_point_select(plane)
                    speed = dest['Speed']

                    if not self.manned_danger_flag:
                        dest = self.safe_bound(plane, self.get_object_by_id(plane['LayoutID']), dest['X'], dest['Y'])
                        if 'Speed' in dest:
                            speed = dest['Speed']

                    if plane['Type'] == 1:
                        dest = self.dest_bound(plane, dest['X'], dest['Y'], 145000, 10000)
                    else:
                        dest = self.rect_dest_bound(dest['X'], dest['Y'], 145000)

                    leader_fire_route_list = [{"X": dest['X'], "Y": dest['Y'], "Z": data['area_max_alt']-500}, ]
                    cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                speed, data['move_max_acc'],
                                                                data["move_max_g"]))
                else:
                    cmd_list.append(
                        CmdEnv.make_areapatrolparam(plane['ID'], plane['X'],
                                                    plane['Y'], data['area_max_alt'], 200, 100,
                                                    data['move_max_speed'], data['move_max_acc'], data['move_max_g']))


    def layout_point_select(self, myself):

        data = self.get_move_data(myself)
        enemy_leftweapon = 0
        for enemy_plane in self.enemy_allplane_infos:
            enemy_leftweapon += enemy_plane['LeftWeapon']
        for enemy_missile in self.enemy_missile_infos:
            if enemy_missile['EngageTargetID'] == myself['ID']:
                enemy_leftweapon += 1
        if enemy_leftweapon <= 1 and myself['Type'] == 1 and myself['LeftWeapon'] == 1:
            return {'X': 0, 'Y': 0, 'Speed': data['move_max_speed']}

        select_plane = self.get_object_by_id(myself['LayoutID'])

        if myself['Type'] == 1:
            if 'EX' not in self.regiments[myself['RegimentID']][0]: 
                return {'X': select_plane['X'], 'Y': select_plane['Y'], 'Speed': data['move_max_speed']}
            enemy_EX = self.regiments[myself['RegimentID']][0]['EX']
            enemy_EY = self.regiments[myself['RegimentID']][0]['EY']
            if 'EX' not in myself:
                if myself['LeftWeapon'] <= 1:
                    dest = self.get_KeepDistance(myself, select_plane, 65000)
                    return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
                if self.is_regiment_win(myself['RegimentID'], False):
                    dest = self.fight_go(myself, select_plane)
                    return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
                else:
                    if (myself['X'] ** 2 + myself['Y'] ** 2) ** 0.5 > 120000:
                        dest = self.fight_go(myself, select_plane)
                        return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
                    else:
                        dest = self.get_KeepDistance(myself, select_plane, 65000)
                        return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}

            my_EX = myself['EX']
            my_EY = myself['EY']
            dx = my_EX - enemy_EX
            dy = my_EY - enemy_EY
            r = (dx ** 2 + dy ** 2) ** 0.5
            if r < 0.0001:
                r = 0.0001
            safe_distance = 5000
            safe_x = enemy_EX + dx * (r + safe_distance) / r
            safe_y = enemy_EY + dy * (r + safe_distance) / r
            if ((safe_x - myself['X']) ** 2 + (safe_y - myself['Y']) ** 2) ** 0.5 > safe_distance:
                return {'X': safe_x, 'Y': safe_y, 'Speed': data['move_max_speed']}
            else:
                return {'X': select_plane['X'], 'Y': select_plane['Y'], 'Speed': data['move_min_speed']}

        else:
            ob_distance = 59500
            if myself['LeftWeapon'] == 0:
                if select_plane['LeftWeapon'] > 2 - select_plane['Type']:
                    my_theta = self.XY2theta(myself['X']-select_plane['X'], myself['Y']-select_plane['Y'])
                    dest_distance = 10000
                    dest_x = myself['X'] + dest_distance * math.sin(my_theta)
                    dest_y = myself['Y'] + dest_distance * math.cos(my_theta)
                    return {'X': dest_x, 'Y': dest_y, 'Speed': data['move_max_speed']}
                if self.enemy_manned_info[0]['LeftWeapon'] > 1:
                    my_theta = self.XY2theta(myself['X']-self.enemy_manned_info[0]['X'], myself['Y']-self.enemy_manned_info[0]['Y'])
                    dest_distance = 10000
                    dest_x = myself['X'] + dest_distance * math.sin(my_theta)
                    dest_y = myself['Y'] + dest_distance * math.cos(my_theta)
                    return {'X': dest_x, 'Y': dest_y, 'Speed': data['move_max_speed']}
                dest = self.get_KeepDistance(myself, select_plane, ob_distance)
                return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
            else:
                if select_plane['Type'] == 1:
                    dest = self.get_KeepDistance(myself, select_plane, 80000)
                    return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
                my_regiment_num = 0
                for my_plane in self.my_allplane_infos:
                    if my_plane['RegimentID'] == myself['RegimentID'] and my_plane['LeftWeapon'] > 2-my_plane['Type']:
                        my_regiment_num += 1
                if my_regiment_num == 1:
                    if 'EX' not in self.regiments[myself['RegimentID']][0]:
                        my_friend_num = 0
                        for my_plane in self.my_allplane_infos:
                            if my_plane['LeftWeapon'] > 2 - my_plane['Type'] and my_plane['ID'] != myself['ID']:
                                my_friend_num += 1
                        if my_friend_num >= 1 and not self.is_revenge_time():
                            dest = self.get_KeepDistance(myself, select_plane, ob_distance)
                            return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
                        else:
                            return {'X': select_plane['X'], 'Y': select_plane['Y'], 'Speed': data['move_max_speed']}
                    else:
                        if self.is_regiment_win(myself['RegimentID'], False):
                            dest = self.fight_go(myself, select_plane)
                            return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
                        else:
                            if (myself['X'] ** 2 + myself['Y'] ** 2) ** 0.5 > 120000:
                                dest = self.fight_go(myself, select_plane)
                                return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
                            else:
                                dest = self.get_KeepDistance(myself, select_plane, ob_distance)
                                return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}
                else:
                    left_num = 0
                    right_num = 0
                    my_theta = self.XY2theta(myself['X'] - select_plane['X'], myself['Y'] - select_plane['Y'])
                    for plane in self.my_uvas_infos:
                        if TSVector3.distance(plane, select_plane) < 0.0001:
                            continue
                        if self.availability(plane) and plane['ID'] != myself['ID'] and plane['RegimentID'] == myself[
                            'RegimentID'] and plane['LeftWeapon'] > 0:  
                            plane_theta = self.XY2theta(plane['X'] - select_plane['X'], plane['Y'] - select_plane['Y'])
                            d_theta = self.pi_bound(plane_theta - my_theta)
                            if d_theta >= 0:
                                right_num += 1
                            else:
                                left_num += 1
                    all_num = right_num + left_num + 1
                    layout_theta_list = [0]
                    if all_num == 2:
                        layout_theta_list = [-65, 65]
                    elif all_num == 3:
                        layout_theta_list = [-65, 0, 65]
                    elif all_num == 4:
                        layout_theta_list = [-65, -30, 30, 65]
                    center_theta = self.XY2theta(myself['EX']-select_plane['X'], myself['EY']-select_plane['Y'])
                    layout_theta = self.pi_bound(center_theta + layout_theta_list[left_num] * math.pi / 180)
                    layout_distance = 65000
                    layout_x = select_plane['X'] + layout_distance * math.sin(layout_theta)
                    layout_y = select_plane['Y'] + layout_distance * math.cos(layout_theta)
                    if TSVector3.distance(myself, select_plane) > layout_distance + 5000:
                        my_distance = TSVector3.distance(myself, select_plane)
                        my_shadow = my_distance * math.cos(self.pi_bound(my_theta - layout_theta))
                        my_dest_distance = (my_shadow + layout_distance) / 2
                        dest_x = select_plane['X'] + my_dest_distance * math.sin(layout_theta)
                        dest_y = select_plane['Y'] + my_dest_distance * math.cos(layout_theta)

                        min_flag = True
                        for my_plane in self.my_uvas_infos:
                            if my_plane['RegimentID'] == myself['RegimentID'] and my_plane['LeftWeapon'] > 2 - my_plane['Type'] and \
                                    my_plane['ID'] != myself['ID']:

                                left_num_ = 0
                                right_num_ = 0
                                my_theta_ = self.XY2theta(my_plane['X'] - select_plane['X'],
                                                         my_plane['Y'] - select_plane['Y'])
                                for plane in self.my_uvas_infos:
                                    if TSVector3.distance(plane, select_plane) < 0.0001:
                                        continue
                                    if self.availability(plane) and plane['ID'] != my_plane['ID'] and plane[
                                        'RegimentID'] == my_plane[
                                        'RegimentID'] and plane['LeftWeapon'] > 0: 
                                        plane_theta_ = self.XY2theta(plane['X'] - select_plane['X'],
                                                                    plane['Y'] - select_plane['Y'])
                                        d_theta_ = self.pi_bound(plane_theta_ - my_theta_)
                                        if d_theta_ >= 0:
                                            right_num_ += 1
                                        else:
                                            left_num_ += 1
                                all_num_ = right_num_ + left_num_ + 1
                                layout_theta_ = self.pi_bound(center_theta + layout_theta_list[left_num_] * math.pi / 180)
                                layout_x_ = select_plane['X'] + layout_distance * math.sin(layout_theta_)
                                layout_y_ = select_plane['Y'] + layout_distance * math.cos(layout_theta_)
                                layout = {'X': layout_x, 'Y': layout_y, 'Z': select_plane['Alt']}
                                layout_ = {'X': layout_x_, 'Y': layout_y_, 'Z': select_plane['Alt']}


                                if TSVector3.distance(my_plane, layout_) < TSVector3.distance(myself, layout):
                                    min_flag = False
                                    break
                        if min_flag:
                            return {'X': dest_x, 'Y': dest_y, 'Speed': data['move_min_speed']}
                        return {'X': dest_x, 'Y': dest_y, 'Speed': data['move_max_speed']}
                    else:
                        my_num = 0
                        for my_plane in self.my_allplane_infos:
                            if my_plane['LeftWeapon'] > 2-my_plane['Type'] and TSVector3.distance(my_plane, select_plane) < 90000 - my_plane['Type'] * 10000:
                                my_num += 1
                        enemy_num = 0
                        for enemy_plane in self.enemy_allplane_infos:
                            if enemy_plane['LeftWeapon'] > 0 and TSVector3.distance(enemy_plane, select_plane) < 60000:
                                enemy_num += 1
                        if my_num >= 2 or my_num >= enemy_num:
                            my_distance = TSVector3.distance(myself, select_plane)
                            my_shadow = my_distance * math.cos(self.pi_bound(my_theta - layout_theta))
                            my_dest_distance = my_shadow / 2
                            dest_x = select_plane['X'] + my_dest_distance * math.sin(layout_theta)
                            dest_y = select_plane['Y'] + my_dest_distance * math.cos(layout_theta)
                            min_flag = True
                            for my_plane in self.my_allplane_infos:
                                if my_plane['LeftWeapon'] > 2-my_plane['Type'] and my_plane['ID'] != myself['ID'] and \
                                        TSVector3.distance(my_plane, select_plane) < TSVector3.distance(myself, select_plane):
                                    min_flag = False
                                    break
                            if min_flag and my_num > 1:
                                return {'X': dest_x, 'Y': dest_y, 'Speed': data['move_min_speed']}
                            else:
                                return {'X': dest_x, 'Y': dest_y, 'Speed': data['move_max_speed']}
                        else:
                            dest = self.get_KeepDistance(myself, select_plane, 65000, layout_theta_list[left_num] * math.pi / 180)
                            return {'X': dest['X'], 'Y': dest['Y'], 'Speed': data['move_max_speed']}



    def get_KeepDistance(self, myself, danger, distance, theta = 4.0): 
        dest_distance = 10000
        toward = 1 
        my_theta = self.XY2theta(myself['X'] - danger['X'], myself['Y'] - danger['Y'])
        my_distance = TSVector3.distance(myself, danger)
        if theta == 4.0:
            left_num = 0
            right_num = 0
            for enemy_plane in self.enemy_allplane_infos:
                if TSVector3.distance(enemy_plane, danger) < 0.0001:
                    continue
                if self.availability(enemy_plane) and enemy_plane['LeftWeapon'] > 0: 
                    enemy_theta = self.XY2theta(enemy_plane['X'] - danger['X'], enemy_plane['Y'] - danger['Y'])
                    d_theta = self.pi_bound(enemy_theta - my_theta)
                    if d_theta >= 0:
                        right_num += 1
                    else:
                        left_num += 1
            if left_num >= right_num:
                toward = -1
        else:
            d_theta = self.pi_bound(theta - my_theta)
            if d_theta <= 0:
                toward = -1
        dest_x = 0
        dest_y = 0
        if my_distance < distance - 1000: 
            dest_x = myself['X'] + math.sin(my_theta) * dest_distance
            dest_y = myself['Y'] + math.cos(my_theta) * dest_distance
        elif my_distance > distance + 1000:
            dest_x = myself['X'] - math.sin(my_theta) * dest_distance
            dest_y = myself['Y'] - math.cos(my_theta) * dest_distance
        else:
            dest_theta = self.pi_bound(my_theta + toward * math.pi / 2)
            dest_x = myself['X'] + math.sin(dest_theta) * dest_distance
            dest_y = myself['Y'] + math.cos(dest_theta) * dest_distance


        return {'X': dest_x, 'Y': dest_y}

    def fight_go(self, myself, danger): 
        dest_distance = 10000
        in_theta = False
        look_theta = math.pi / 3
        if danger['Type'] == 2:
            look_theta /= 2
        my_theta = self.XY2theta(myself['X'] - danger['X'], myself['Y'] - danger['Y'])
        d_theta = self.pi_bound(my_theta - danger['Heading'])
        if abs(d_theta) < look_theta:
            in_theta = True

        dest_x = myself['X'] - math.sin(my_theta) * dest_distance
        dest_y = myself['Y'] - math.cos(my_theta) * dest_distance
        if not self.is_looking(danger, myself, danger['Heading']) and in_theta and danger['LeftWeapon'] > 0: 
            toward = 1 
            if d_theta > 0:
                toward = -1
            dest_theta = self.pi_bound(-my_theta)
            my_look_theta = math.pi / 3
            if myself['Type'] == 2:
                my_look_theta /= 2
            dest_theta = self.pi_bound(dest_theta + toward * (my_look_theta - math.pi / 18))
            dest_x = myself['X'] + math.sin(dest_theta) * dest_distance
            dest_y = myself['Y'] + math.cos(dest_theta) * dest_distance
        return {'X': dest_x, 'Y': dest_y}

    def safe_bound(self, myself, enemy, dest_x, dest_y):
        safe_distance = 65000
        dest_distance = 10000
        left_num = 0
        right_num = 0
        for enemy_plane in self.regiments[enemy['RegimentID']]:
            if TSVector3.distance(enemy_plane, enemy) < 0.0001:
                continue
            if self.availability(enemy_plane) and enemy_plane['LeftWeapon'] > 0: 
                my_theta = self.XY2theta(myself['X'] - dest_x, myself['Y'] - dest_y)
                enemy_theta = self.XY2theta(enemy_plane['X'] - enemy['X'], enemy_plane['Y'] - enemy['Y'])
                d_theta = self.pi_bound(enemy_theta - my_theta)
                if d_theta >= 0:
                    right_num += 1
                else:
                    left_num += 1
        toward = 1 
        if left_num >= right_num:
            toward = -1
        dest_x_ = 0
        dest_y_ = 0
        for enemy_plane in self.enemy_allplane_infos:
            if enemy_plane['ID'] != enemy['ID'] and enemy_plane['LeftWeapon'] > 0:
                if TSVector3.distance(enemy_plane, myself) < safe_distance:
                    my_theta = self.XY2theta(myself['X'] - enemy_plane['X'], myself['Y'] - enemy_plane['Y'])
                    dest_enemy_x = enemy_plane['X'] + (safe_distance + 5000) * math.sin(my_theta)
                    dest_enemy_y = enemy_plane['Y'] + (safe_distance + 5000) * math.cos(my_theta)
                    dest_x_ += dest_enemy_x - myself['X']
                    dest_y_ += dest_enemy_y - myself['Y']
                if TSVector3.distance(enemy_plane, myself) < safe_distance + 5000:
                    my_theta = self.XY2theta(myself['X'] - enemy_plane['X'], myself['Y'] - enemy_plane['Y'])
                    dest_enemy_theta = self.pi_bound(my_theta + toward * math.pi / 30)
                    dest_enemy_x = enemy_plane['X'] + (safe_distance + 5000) * math.sin(dest_enemy_theta)
                    dest_enemy_y = enemy_plane['Y'] + (safe_distance + 5000) * math.cos(dest_enemy_theta)
                    dest_x_ += dest_enemy_x - myself['X']
                    dest_y_ += dest_enemy_y - myself['Y']
        r = (dest_x_ ** 2 + dest_y_ ** 2) ** 0.5
        if r < 0.0001:
            dest_x_ = dest_x
            dest_y_ = dest_y
            dest = {'X': dest_x_, 'Y': dest_y_}
        else:
            dest_x_ = myself['X'] + dest_x_ * dest_distance / r
            dest_y_ = myself['Y'] + dest_y_ *  dest_distance / r
            data = self.get_move_data(myself)
            dest = {'X': dest_x_, 'Y': dest_y_, 'Speed': data['move_max_speed']}
        return dest



    def is_regiment_win(self, regiment_ID, flag = False): 
        my_win = 0
        for my_plane in self.my_allplane_infos:
            if my_plane['RegimentID'] == regiment_ID:
                if my_plane['LeftWeapon'] == 0:
                    my_win += 0.1
                if my_plane['LeftWeapon'] >= 1:
                    my_win += 1 + (my_plane['LeftWeapon'] - 1) * 0.2
        enemy_win = 0
        for enemy_plane in self.enemy_allplane_infos:
            if enemy_plane['RegimentID'] == regiment_ID:
                if enemy_plane['LeftWeapon'] == 0:
                    enemy_win += 0.1
                if enemy_plane['LeftWeapon'] >= 1:
                    enemy_win += 1 + (enemy_plane['LeftWeapon'] - 1) * 0.2
        if flag:
            return my_win > enemy_win
        return my_win >= enemy_win

    def is_in_cold(self, item0, item1):
        cold_time = 10
        min_time = 1200
        for missile in self.my_missile_infos + self.enemy_missile_infos:
            if missile['LauncherID'] == item0['ID'] and missile['EngageTargetID'] == item1['ID'] and item0['CurTime']-missile['InitTime'] < min_time:
                min_time = item0['CurTime']-missile['InitTime']
        if min_time < cold_time:
            return True
        return False

    def is_guidanced(self, myself):
        for missile_ID in myself['LockedMissileID']:
            missile = self.get_object_by_id(missile_ID)
            if missile is not None and TSVector3.distance(myself, missile) > self.terminal_guidance_distance:
                return True
        return False

    def is_revenge_time(self): 
        my_win = len(self.my_uvas_infos) * 15
        enemy_win = len(self.enemy_uvas_infos) * 15
        for my_plane in self.my_allplane_infos:
            my_win += my_plane['LeftWeapon'] * 3
        for enemy_plane in self.enemy_allplane_infos:
            enemy_win += enemy_plane['LeftWeapon'] * 3
        return self.my_manned_info[0]['CurTime'] >= 20 * 60 -300 and (
                my_win < enemy_win or my_win == enemy_win and self.my_score < self.enemy_score)


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
                            theta = self.pi_bound(theta - math.pi/2)
                        else:
                            theta = self.pi_bound(theta + math.pi/2)
                        s_dict[plane0['ID']][plane1['ID']]['dx'] = math.sin(theta)
                        s_dict[plane0['ID']][plane1['ID']]['dy'] = math.cos(theta)
        return s_dict


    def get_danger_speed(self, item):
        danger_move_v = 1000
        if item['Type'] == 3 and item['CurTime'] - self.missile_init_time[item['ID']] > 30:  
            danger_move_v = item['Speed']
        return danger_move_v


    def can_escape(self, item0, item1, theta):
        distance = TSVector3.distance(item0, item1)
        if item0['Type'] == 3:
            data = self.get_move_data(item1)
            start_time = item0['CurTime'] - item0['InitTime']
            rest_time = 120-start_time
            my_turn_t = self.get_escape_turn_time(item1, theta, 0)
            rest_time -= my_turn_t
            if rest_time < 0:
                rest_time = 0
            plane_distance = data['move_max_speed'] * rest_time
            missile_plane_theta = self.XY2theta(item1['X']-item0['X'], item1['Y']-item0['Y'])
            missile_distance = ((TSVector3.distance(item0, item1) * math.sin(missile_plane_theta) + plane_distance * math.sin(theta))**2 + \
                                (TSVector3.distance(item0, item1) * math.cos(missile_plane_theta) + plane_distance * math.cos(theta))**2)**0.5


            if missile_distance > self.missile_path_distance[120] - self.missile_path_distance[int(start_time)] + 200:
                return True
            return False
        else:
            enemy_turn_t = self.get_fire_turn_time(item0, item1, 1)
            start_time = 0
            shot_time = 0
            for i in range(int(start_time), 122):
                if i == 121:
                    shot_time = 1200
                    break
                if self.missile_path_distance[i] - self.missile_path_distance[int(start_time)] + 200 >= distance:
                    shot_time = i - start_time + enemy_turn_t
                    break
            if shot_time > self.get_escape_turn_time(item1, self.XY2theta(item0['X']-item1['X'], item0['Y']-item1['Y']), 0):
                return True
            return False

    def get_tangent_shot_time(self, item0, item1): 
        distance = TSVector3.distance(item0, item1)
        start_time = 0
        enemy_turn_t = 0
        if item0['Type'] != 3:
            enemy_turn_t = self.get_fire_turn_time(item0, item1, 1) 
        else:
            start_time = item0['CurTime'] - item0['InitTime']
        for i in range(int(start_time), 121):
            if self.missile_path_distance[i] - self.missile_path_distance[int(start_time)] + 200 >= distance:
                return i - start_time + enemy_turn_t
        return 1200

    def get_danger_shot_time(self, item0, item1, theta):
        data = self.get_move_data(item1)
        my_turn_t = self.get_escape_turn_time(item1, theta, 0)
        start_time = 0
        enemy_turn_t = 0
        if item0['Type'] != 3:
            enemy_turn_t = self.get_fire_turn_time(item0, item1, 1) 
        else:
            start_time = item0['CurTime'] - item0['InitTime']
        for i in range(int(start_time), 121):
            rest_time = i - start_time
            rest_time += enemy_turn_t - my_turn_t
            if rest_time < 0:
                rest_time = 0
            plane_distance = data['move_max_speed'] * rest_time
            missile_plane_theta = self.XY2theta(item1['X'] - item0['X'], item1['Y'] - item0['Y'])
            missile_distance = ((TSVector3.distance(item0, item1) * math.sin(missile_plane_theta) + plane_distance * math.sin(theta)) ** 2 +
                                (TSVector3.distance(item0, item1) * math.cos(missile_plane_theta) + plane_distance * math.cos(theta)) ** 2) ** 0.5
            if self.missile_path_distance[i] - self.missile_path_distance[int(start_time)] + 200 >= missile_distance:
                return i - start_time + enemy_turn_t
        return 1200


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

            r = (plane['X']**2 + plane['Y']**2) ** 0.5
            if r > (150000 + max_bound)/2: 
                d_theta = self.pi_bound(plane['Heading'] - self.XY2theta(plane['X'], plane['Y']))
                toward = 1
                if d_theta < 0:
                    toward = -1
                dest_theta = self.pi_bound(self.XY2theta(plane['X'], plane['Y']) + toward * math.pi / 18)
                old_dest_theta = self.XY2theta(dest['X']-plane['X'], dest['Y']-plane['Y'])
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

    def rect_dest_bound(self, dest_x, dest_y, max_bound):
        dest = {'X': dest_x, 'Y': dest_y}
        if dest_x < -max_bound:
            dest['X'] = -max_bound
        if dest_x > max_bound:
            dest['X'] = max_bound
        if dest_y < -max_bound:
            dest['Y'] = -max_bound
        if dest_y > max_bound:
            dest['Y'] = max_bound
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


    def get_object_by_id(self, objectID) -> {}:
        for plane in self.my_allplane_infos + self.enemy_allplane_infos:
            if plane['ID'] == objectID:
                return plane
        for missile in self.my_missile_infos + self.enemy_missile_infos:
            if missile['ID'] == objectID:
                return missile
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
                                 90780.74065129971, 91310.91585482856, 91838.29318928291, 91838.29318928291+525.8610703383404]

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

    def equal(self, obj1, obj2):
        return abs(obj1 - obj2) < 0.0001