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

class Modify(Agent):
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
        super(Modify, self).__init__(name, config["side"])
        self._init() #调用用以定义一些下文所需变量

    def _init(self):
        """对下文中使用到的变量进行定义和初始化"""
        self.my_uvas_infos = []         #该变量用以保存己方所有无人机的态势信息
        self.my_manned_info = []        #该变量用以保存己方有人机的态势信息
        self.my_allplane_infos = []     #该变量用以保存己方所有飞机的态势信息
        self.my_missile_infos = []      # ××× 该变量用以保存我方导弹的态势信息
        self.enemy_uvas_infos = []      #该变量用以保存敌方所有无人机的态势信息
        self.enemy_manned_info = []     #该变量用以保存敌方有人机的态势信息
        self.enemy_allplane_infos = []  #该变量用以保存敌方所有飞机的态势信息
        self.enemy_missile_infos = []   #该变量用以保存敌方导弹的态势信息
        self.enemy_leftweapon = {}      # ××× 该变量用以保存敌方飞机的导弹量
        self.regiments = []             # *** 团战

        self.attack_handle_enemy = {}   # ××× 该变量用于记录已经去攻击的飞机

        self.missile_path_length = {} # ××× 导弹的已飞路程
        self.missile_last_pisition = {} # ××× 导弹上个时刻的位置
        self.missile_init_time = {} # ××× 导弹的初始化时间
        self.attacked_flag = {}  # ××× 判断飞机是否被攻击 飞机ID检索 元素是列表 保存所有攻击该飞机的导弹的ID
        self.curtime_has_attack = {} # *** 当前秒发射导弹信息

        self.safe_flag = False
        self.attack_dict = {}

        self.escape_toward = {}  # 历史切向选择信息
        self.escape_last_dangerID = {}  # 历史威胁者

        self.missile_to_enemy_manned = 0
        self.missile_to_enemy_uvas = 0

        self.my_score = 0
        self.enemy_score = 0

        self.my_allplane_maps = {}

        self.missile_path_distance = self.get_missile_data()['path_distance']
        self.missile_v = self.get_missile_data()['v']
        self.terminal_guidance_distance = 20000

    def reset(self, **kwargs):
        """当引擎重置会调用,选手需要重写此方法,来实现重置的逻辑"""
        self.my_uvas_infos.clear()  # 该变量用以保存己方所有无人机的态势信息
        self.my_manned_info.clear()  # 该变量用以保存己方有人机的态势信息
        self.my_allplane_infos.clear()  # 该变量用以保存己方所有飞机的态势信息
        self.my_missile_infos.clear()   #该变量用以保存我方导弹的态势信息
        self.enemy_uvas_infos.clear()  # 该变量用以保存敌方所有无人机的态势信息
        self.enemy_manned_info.clear()  # 该变量用以保存敌方有人机的态势信息
        self.enemy_allplane_infos.clear()  # 该变量用以保存敌方所有飞机的态势信息
        self.enemy_missile_infos.clear()  # 该变量用以保存敌方导弹的态势信息
        self.enemy_leftweapon.clear()  # 该变量用以保存敌方飞机的导弹量

        self.attack_handle_enemy.clear()  # 该变量用于记录已经去攻击的飞机

        self.missile_path_length.clear()   #导弹的已飞路程
        self.missile_last_pisition.clear() #导弹上个时刻的位置
        self.missile_init_time.clear()     #导弹的初始化时间
        self.attacked_flag.clear()         # 判断飞机是否被攻击 飞机ID检索 元素是列表 保存所有攻击该飞机的导弹的ID
        self.curtime_has_attack.clear() # *** 当前秒发射导弹信息

        self.safe_flag = False
        self.attack_dict.clear()

        self.escape_toward.clear()  # 历史切向选择信息
        self.escape_last_dangerID.clear()  # 历史威胁者

        self.missile_to_enemy_manned = 0
        self.missile_to_enemy_uvas = 0

        self.my_score = 0
        self.enemy_score = 0

        self.my_allplane_maps.clear()

        pass

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        """ 步长处理
        此方法继承自基类中的step(self,sim_time, obs_red, **kwargs)
        选手通过重写此方法，去实现自己的策略逻辑，注意，这个方法每个步长都会被调用
        :param sim_time: 当前想定已经运行时间
        :param obs_side:当前方所有的态势信息信息，包含了所有的当前方所需信息以及探测到的敌方信息
				obs_side 包含 platforminfos，trackinfos，missileinfos三项Key值
				obs_side['platforminfos'] 为己方飞机信息，字典列表，其中包含的字典信息如下（以下为Key值）

                        Name 			# 飞机的名称
                        Identification 	# 飞机的标识符（表示飞机是红方还是蓝方）
                        ID 				# 飞机的ID（表示飞机的唯一编号）
                        Type 			# 飞机的类型（表示飞机的类型，其中有人机类型为 1，无人机类型为2）
						Availability 	# 飞机的可用性（表示飞机的可用性，范围为0到1,为1表示飞机存活，0表示飞机阵亡）
						X 				# 飞机的当前X坐标（表示飞机的X坐标）
						Y 				# 飞机的当前Y坐标（表示飞机的Y坐标）
						Lon 			# 飞机的当前所在经度（表示飞机的所在经度）
						Lat 			# 飞机的当前所在纬度（表示飞机的所在纬度）
						Alt 			# 飞机的当前所在高度（表示飞机的所在高度）
						Heading 		# 飞机的当前朝向角度（飞机的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 飞机的当前俯仰角度（飞机的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 飞机的当前滚转角度（飞机的当前滚转,范围为-180°到180° ）
						Speed 			# 飞机的当前速度（飞机的当前速度）
						CurTime 		# 当前时间（当前时间）
						AccMag 			# 飞机的指令加速度（飞机的指令加速度）
						NormalG 		# 飞机的指令过载（飞机的指令过载）
						IsLocked 		# 飞机是否被敌方导弹锁定（飞机是否被敌方导弹锁定）
						Status 			# 飞机的当前状态（飞机的当前状态）
						LeftWeapon 		# 飞机的当前剩余导弹数（飞机的当前剩余导弹数）

						**LockedMissileID # EngageTargetID为自己的导弹
						**RegimentID    # 要参与的团战ID


				obs_side['trackinfos'] 为敌方飞机信息，字典列表，其中包含的字典信息如下（以下为Key值）

						Name 			# 敌方飞机的名称
						Identification 	# 敌方飞机的标识符（表示敌方飞机是红方还是蓝方）
						ID 				# 敌方飞机的ID（表示飞机的唯一编号）
						Type 			# 敌方飞机的类型（表示飞机的类型，其中有人机类型为 1，无人机类型为2）
						Availability 	# 敌方飞机的可用性（表示飞机的可用性，范围为0到1,为1表示飞机存活，0表示飞机阵亡）
						X 				# 敌方飞机的当前X坐标（表示飞机的X坐标）
						Y 				# 敌方飞机的当前Y坐标（表示飞机的Y坐标）
						Lon 			# 敌方飞机的当前所在经度（表示飞机的所在经度）
						Lat 			# 敌方飞机的当前所在纬度（表示飞机的所在纬度）
						Alt 			# 敌方飞机的当前所在高度（表示飞机的所在高度）
						Heading 		# 敌方飞机的当前朝向角度（飞机的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 敌方飞机的当前俯仰角度（飞机的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 敌方飞机的当前滚转角度（飞机的当前滚转,范围为-180°到180° ）
						Speed 			# 敌方飞机的当前速度（飞机的当前速度）
						CurTime 		# 当前时间（当前时间）
						IsLocked 		# 敌方飞机是否被敌方导弹锁定（飞机是否被己方导弹锁定）

						**LeftWeapon    # 飞机的当前剩余导弹数（飞机的当前剩余导弹数）
						**LockedMissileID # EngageTargetID为自己的导弹
						**RegimentID    # 处于的团战ID

				obs_side['missileinfos']为空间中所有未爆炸的双方导弹信息，字典列表，其中包含的字典信息如下（以下为Key值）

						Name 			# 导弹的名称
						Identification 	# 导弹的标识符（表示导弹是红方还是蓝方）
						ID 				# 导弹的ID（表示导弹的唯一编号）
						Type 			# 导弹的类型（表示导弹的类型，其中导弹类型为 3）
						Availability 	# 导弹的可用性（表示导弹的可用性，范围为0到1,为1表示飞机存活，0表示导弹已爆炸）
						X 				# 导弹的当前X坐标（表示导弹的X坐标）
						Y 				# 导弹的当前Y坐标（表示导弹的Y坐标）
						Lon 			# 导弹的当前所在经度（表示导弹的所在经度）
						Lat 			# 导弹的当前所在纬度（表示导弹的所在纬度）
						Alt 			# 导弹的当前所在高度（表示导弹的所在高度）
						Heading 		# 导弹的当前朝向角度（导弹的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 导弹的当前俯仰角度（导弹的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 导弹的当前滚转角度（导弹的当前滚转,范围为-180°到180° ）
						Speed 			# 导弹的当前速度（导弹的当前速度）
						CurTime 		# 当前时间（当前时间）
						LauncherID 		# 导弹的发射者ID（敌方导弹的发射者ID）
						EngageTargetID 	# 导弹攻击目标的ID（敌方导弹攻击目标的ID）

						**InitTime      # 导弹的初始化时间

        :param kwargs:保留的变量
        :return: 决策完毕的任务指令列表
        """
        cmd_list = []   # 此变量为 保存所有的决策完毕任务指令列表

        self.process_decision(sim_time, obs_side, cmd_list) # 调用决策函数进行决策判断
            

        return cmd_list # 返回决策完毕的任务指令列表

    def process_decision(self, sim_time, obs_side, cmd_list):
        """处理决策
        :param sim_time: 当前想定已经运行时间
        :param obs_side: 当前方所有的Observation信息，包含了所有的当前方所需信息以及探测到的敌方信息
        :param cmd_list保存所有的决策完毕任务指令列表
				可用指令有六种
					1.初始化实体指令 （初始化实体的信息，注意该指令只能在开始的前3秒有效）
						make_entityinitinfo(receiver: int,x: float,y: float,z: float,init_speed: float,init_heading: float)
						参数含义为
							:param receiver:飞机的唯一编号，即上文中飞机的ID
							:param x: 初始位置为战场x坐标
							:param y: 初始位置为战场y坐标
							:param z: 初始位置为战场z坐标
							:param init_speed: 初始速度(单位：米/秒，有人机取值范围：[150,400]，无人机取值范围：[100,300])
							:param init_heading: 初始朝向(单位：度，取值范围[0,360]，与正北方向的夹角)
					2.航线巡逻控制指令（令飞机沿航线机动）
						make_linepatrolparam(receiver: int,coord_list: List[dict],cmd_speed: float,cmd_accmag: float,cmd_g: float)
						参数含义为
							:param receiver: 飞机的唯一编号，即飞机的ID
							:param coord_list: 路径点坐标列表 -> [{"x": 500, "y": 400, "z": 2000}, {"x": 600, "y": 500, "z": 3000}]
											   区域x，y不得超过作战区域,有人机高度限制[2000,15000]，无人机高度限制[2000,10000]
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					3.区域巡逻控制指令	（令飞机沿区域巡逻）
						make_areapatrolparam(receiver: int,x: float,y: float,z: float,area_length: float,area_width: float,cmd_speed: float,cmd_accmag: float,cmd_g: float)
						    :param receiver: 飞机的唯一编号，即飞机的ID
							:param x: 区域中心坐标x坐标
							:param y: 区域中心坐标y坐标
							:param z: 区域中心坐标z坐标
							:param area_length: 区域长
							:param area_width: 区域宽
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					4.机动参数调整控制指令（调整飞机的速度、加速度和过载）
						make_motioncmdparam(receiver: int, update_motiontype: int,cmd_speed: float,cmd_accmag: float,cmd_g: float)
						    :param receiver: 飞机的唯一编号，即飞机的ID
							:param update_motiontype: 调整机动参数,其中 1为设置指令速度，2为设置指令加速度，3为设置指令速度和指令加速度
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					5.跟随目标指令 （令飞机跟随其他飞机）
						make_followparam(receiver: int,tgt_id: int,cmd_speed: float,cmd_accmag: float,cmd_g: float)
							:param receiver:飞机的唯一编号，即飞机的ID
							:param tgt_id: 目标ID,友方敌方均可
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					6.打击目标指令（令飞机使用导弹攻击其他飞机）
						make_attackparam(receiver: int,tgt_id: int,fire_range: float )
							:param receiver:飞机的唯一编号，即飞机的ID
							:param tgt_id: 目标ID
							:param fire_range: 开火范围，最大探测范围的百分比，取值范围[0, 1]

        """
        self.process_observation(obs_side)  # 获取态势信息,对态势信息进行处理

        if sim_time == 1:  # 当作战时间为1s时,初始化实体位置,注意,初始化位置的指令只能在前三秒内才会被执行
            self.init_pos(cmd_list) # 将实体放置到合适的初始化位置上

        if sim_time >= 2:  # 当作战时间大于10s时,开始进行任务控制,并保存任务指令;
            self.attack_dict = self.attack_select() # 处理攻击，己方使用导弹打击敌方飞机
            self.process_move(sim_time, cmd_list)   # 处理机动，己方如何机动

    def process_observation(self, obs_side):
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

        for enemy_plane in enemy_allplane_infos: # 初始化敌方飞机当前载弹量
            if enemy_plane['ID'] not in self.enemy_leftweapon:
                if enemy_plane['Type'] == 1:
                    self.enemy_leftweapon[enemy_plane['ID']] = 4
                else:
                    self.enemy_leftweapon[enemy_plane['ID']] = 2

        for missile in my_missile_infos + enemy_missile_infos: # 对导弹信息进行统计
            old_missile_ID_list = []
            for old_missile in self.my_missile_infos + self.enemy_missile_infos:
                old_missile_ID_list.append(old_missile['ID'])
            if missile['ID'] not in old_missile_ID_list: #出现新导弹
                self.missile_path_length[missile['ID']] = 0 #初始化已走导弹路径长度
                self.missile_last_pisition[missile['ID']] = {'X': 0, 'Y': 0, 'Z': 0}  #初始化上时刻位置
                self.missile_init_time[missile['ID']] = missile['CurTime'] #统计导弹发射时间
                if missile['LauncherID'] in self.enemy_leftweapon: #更新敌方飞机的当前载弹量
                    self.enemy_leftweapon[missile['LauncherID']] -= 1
            else:
                self.missile_path_length[missile['ID']] += TSVector3.distance(
                    self.missile_last_pisition[missile['ID']], missile) #更新已走导弹路径长度
            self.missile_last_pisition[missile['ID']]['X'] = missile['X'] #更新上时刻位置
            self.missile_last_pisition[missile['ID']]['Y'] = missile['Y']
            self.missile_last_pisition[missile['ID']]['Z'] = missile['Z']


        #更新飞机被攻击状态
        for plane in my_allplane_infos + enemy_allplane_infos:

            if plane['ID'] not in self.attacked_flag: # 初始化飞机被攻击标志
                self.attacked_flag[plane['ID']] = []

            availability_missile_ID_list = []

            for missile in my_missile_infos + enemy_missile_infos: #新增的攻击该机的导弹
                availability_missile_ID_list.append(missile['ID'])
                if plane['ID'] == missile['EngageTargetID'] and missile['ID'] not in self.attacked_flag[plane['ID']]:
                    self.attacked_flag[plane['ID']].append(missile['ID'])

            remove_num = 0 #删除失效的导弹
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

        self.curtime_has_attack = {}                    # 清空当前秒导弹发射信息
        for enemy in enemy_allplane_infos:
            self.curtime_has_attack[enemy['ID']] = 0

        for my_manned in my_manned_info:
            if self.is_in_center(my_manned):
                self.my_score += 1
        for enemy_manned in enemy_manned_info:
            if self.is_in_center(enemy_manned):
                self.enemy_score += 1

        self.my_uvas_infos = my_uvas_infos              # 保存当前己方无人机信息
        self.my_manned_info = my_manned_info            # 保存当前己方有人机信息
        self.my_allplane_infos = my_allplane_infos            # 保存当前己方所有飞机信息
        self.my_missile_infos = my_missile_infos     # 保存己方已发射且尚未爆炸的导弹信息
        self.enemy_uvas_infos = enemy_uvas_infos        # 保存当前敌方无人机信息
        self.enemy_manned_info = enemy_manned_info      # 保存当前敌方有人机信息
        self.enemy_allplane_infos = enemy_allplane_infos      # 保存当前敌方所有飞机信息
        self.enemy_missile_infos = enemy_missile_infos  # 保存敌方已发射且尚未爆炸的导弹信息

        # 相关信息导入进关键字
        for enemy_plane_ID in self.enemy_leftweapon: #飞机的剩余导弹
            enemy_plane = self.get_object_by_id(enemy_plane_ID)
            if enemy_plane is not None:
                enemy_plane['LeftWeapon'] = self.enemy_leftweapon[enemy_plane_ID]

        for plane_ID in self.attacked_flag: #飞机的被锁定
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
            if len(self.enemy_manned_info) > 0 and self.enemy_manned_info[0] in self.regiments[i]:
                t = self.regiments[i]
                self.regiments[i] = self.regiments[-1]
                self.regiments[-1] = t
                break
        if len(self.enemy_manned_info) > 1:
            for i in range(len(self.regiments)):
                if self.enemy_manned_info[1] in self.regiments[i]:
                    t = self.regiments[i]
                    idx = -2 if self.enemy_manned_info[0] in self.regiments[-1] else -1
                    self.regiments[i] = self.regiments[idx]
                    self.regiments[idx] = t
                    break
        for i in range(len(self.regiments)-2): 
            for j in range(len(self.regiments)-i-2):
                if len(self.regiments[j]) > len(self.regiments[j+1]):
                    t = self.regiments[j]
                    self.regiments[j] = self.regiments[j+1]
                    self.regiments[j+1] = t

        '''
        for regiment in self.regiments:
            for plane in regiment:
                print(plane['Name'])
            print('-------')
        print('######')
        '''
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
            elif len(self.my_manned_info) > 1 and self.my_manned_info[1]['RegimentID'] == -1:
                self.my_manned_info[1]['RegimentID'] = 0
                i += 1
            regiment = self.regiments[regiment_ID]
            while i <= len(regiment): #
                min_distance = 450000
                min_uvas = {}
                for my_uvas in self.my_uvas_infos:
                    if my_uvas['RegimentID'] == -1 and my_uvas['LeftWeapon'] > 0: #
                        for enemy_plane in self.regiments[0]:
                            if TSVector3.distance(enemy_plane, my_uvas) < min_distance:
                                min_distance = TSVector3.distance(enemy_plane, my_uvas)
                                min_uvas = my_uvas
                if len(min_uvas) == 0:
                    for my_uvas in self.my_uvas_infos:
                        if my_uvas['RegimentID'] == -1 and my_uvas['LeftWeapon'] == 0: #
                            for enemy_plane in self.regiments[0]:
                                if TSVector3.distance(enemy_plane, my_uvas) < min_distance:
                                    min_distance = TSVector3.distance(enemy_plane, my_uvas)
                                    min_uvas = my_uvas
                if len(min_uvas) == 0:
                    break
                min_uvas['RegimentID'] = regiment_ID
                i += 1
        for my_plane in self.my_allplane_infos:
            if my_plane['RegimentID'] == -1:
                my_plane['RegimentID'] = 0
        '''
        print(self.my_manned_info[0]['CurTime'])
        for my_plane in self.my_allplane_infos:
            print(my_plane['Name'], my_plane['RegimentID'])
        for enemy_plane in self.enemy_allplane_infos:
            print(enemy_plane['Name'], enemy_plane['RegimentID'])
        '''
        self.safe_flag = self.is_safe()

    def init_pos(self, cmd_list):
        """
        初始化飞机部署位置
        :param cmd_list:所有的决策完毕任务指令列表
        """
        leader_original_pos = {}   
        leader_heading = 0
        if self.name == "red":
            leader_original_pos = {"X": -125000, "Y": -100000, "Z": 9000}
            leader_heading = 90
        else :
            leader_original_pos = {"X": 125000, "Y": 100000, "Z": 9000}
            leader_heading = 270

        interval_distance = 5000   
        init_distance = interval_distance / 2
        if len(self.my_manned_info) == 2:           # 初始化时一定有两架有人机
            cmd_list.append(CmdEnv.make_entityinitinfo(self.my_manned_info[0]['ID'], leader_original_pos['X'], leader_original_pos['Y'] + init_distance, leader_original_pos['Z'],400, leader_heading))
            cmd_list.append(CmdEnv.make_entityinitinfo(self.my_manned_info[1]['ID'], leader_original_pos['X'], leader_original_pos['Y'] - init_distance, leader_original_pos['Z'],400, leader_heading))

      
        sub_index = 0  
        for sub in self.my_uvas_infos: 
            sub_pos = copy.deepcopy(leader_original_pos)  
            if sub_index & 1 == 0: 
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] + init_distance + interval_distance, sub_pos['Z'],300 * 0.6, leader_heading))
            else:                   # 将当前编号放在有人机的另一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] - init_distance - interval_distance, sub_pos['Z'],300 * 0.6, leader_heading))
           
                interval_distance *= 2
            sub_index += 1


    def attack_select(self):
        """
        :param myself: 主参考对象
        :return:攻击字典, key为plane['ID'], value为列表[plane['ID'], ]
        """
        can_attack_dict = {}
        for enemy_plane in self.enemy_allplane_infos:
            can_attack_dict[enemy_plane['ID']] = []

        curtime_attack_myplane = {}
        for my_plane in self.my_allplane_infos:
            curtime_attack_myplane[my_plane['ID']] = False

  

        danger_enemy_uvas_num = 0
        for enemy_uvas in self.enemy_uvas_infos:
            if enemy_uvas['LeftWeapon'] > 0:
                danger_enemy_uvas_num += 1
        if danger_enemy_uvas_num > 0:
            # my_leader = self.my_manned_info[0]
            for my_leader in self.my_manned_info:
                for plane in self.my_allplane_infos:
                    for enemy_plane in self.enemy_allplane_infos:
                        if TSVector3.distance(enemy_plane, my_leader) < 15000 and len(
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
                                if min_flag: 
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

        for enemy_leader in self.enemy_uvas_infos:


            max_distance = -1 
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
                    if len(trigger_plane) != 0 and trigger_plane['ID'] == my_plane['ID']: 
                        fire_flag = True
                        fire_type = 1
                    if not fire_flag and TSVector3.distance(my_plane, enemy_leader) < 20000:
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
                    if fire_flag and self.safe_flag:
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
        if len(self.enemy_uvas_infos) == 0 and len(self.enemy_missile_infos) == 0:
            for enemy_manned in self.enemy_manned_info:
                if enemy_manned['LeftWeapon'] > 1:
                    return False 
            return True
        return False


    def process_move(self, sim_time, cmd_list):

        for plane in self.my_allplane_infos:
            self.single_process_move(sim_time, cmd_list, plane)


    def single_process_move(self, sim_time, cmd_list, plane):


        if len(self.my_manned_info) > 0:

            for enemy_ID in self.attack_dict.keys(): #攻击
                enemy_plane = self.get_object_by_id(enemy_ID)
                if plane['ID'] in self.attack_dict[enemy_ID] and \
                        (self.is_looking(plane, enemy_plane, plane['Heading']) or
                         enemy_plane['Type'] == 1 and plane['Type'] == 2):
                    cmd_list.append(CmdEnv.make_attackparam(plane['ID'], enemy_ID, 1))
                    return

            dest_distance = 10000 
            max_bound = 145000  
            # enemy_manned = self.enemy_manned_info[0]
       
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
                            self.get_escape_turn_time(plane, self.XY2theta(missile['X']-plane['X'], missile['Y']-plane['Y']), 0)): #随便飞都打不到时
                        continue
                    missile_time = self.get_tangent_shot_time(missile, plane)
                    danger_list.append(missile)
                    if missile_time < lock_min_time:
                        lock_min_distacne = distance
                        lock_min_time = missile_time
                        lock_min_danger = missile

            for enemy_plane in self.enemy_allplane_infos: 
                if self.availability(enemy_plane) and (enemy_plane['LeftWeapon'] > 0 and plane['Type'] == 1 or
                                                       enemy_plane['LeftWeapon'] > 2-enemy_plane['Type'] and plane['Type'] == 2): #有导弹的敌机
                    distance = TSVector3.distance(plane, enemy_plane)
                    if distance < danger_tangent_distance and plane['Type'] == 1 or \
                            distance < danger_uvas_tangent_distance and plane['Type'] == 2: 
                        enemy_time = self.get_tangent_shot_time(enemy_plane, plane) 
                        danger_list.append(enemy_plane)
                        if enemy_time < lock_min_time:
                            lock_min_distacne = distance
                            lock_min_time = enemy_time
                            lock_min_danger = enemy_plane
            '''
            if len(danger_list) > 0:
                if lock_min_danger['Type'] == 3:
                    print(lock_min_danger['Name'])
                    print('distance: ', TSVector3.distance(plane, lock_min_danger),
                          ' danger_v: ', lock_min_danger['Speed'],
                          ' plane_v: ', plane['Speed'])
                    print('missile_path_length: ', self.missile_path_length,
                          'missile_situ', [lock_min_danger['Heading']*180/math.pi, lock_min_danger['Pitch'], lock_min_danger['Roll']],
                          'plane_situ', [plane['Heading']*180/math.pi, plane['Pitch'], plane['Roll']])
            '''

         
            if len(danger_list) > 0 and not (self.safe_flag and plane['Type'] == 2):
                tangent_flag = False
                dx = 0
                dy = 0
                if plane['Type'] == 1:
                    #if lock_min_distacne < danger_tangent_distance and
                    if self.can_escape(lock_min_danger, plane,
                            self.XY2theta(plane['X'] -lock_min_danger['X'], plane['Y']-lock_min_danger['Y'])) is False and \
                            lock_min_danger['Type'] != 1:
                        tangent_flag = True
                        if lock_min_danger['Type'] == 3 and lock_min_time + lock_min_danger['CurTime'] - lock_min_danger['InitTime'] > 84: #不可切向规避
                      
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

                
                if abs(track_theta) >= looking_theta * 0.9: #当前过界
                    only_track_flag = 1
                if abs(self.pi_bound(track_theta - self.pi_bound(enemy_w_tangent * 5))) >= looking_theta * 0.9: #5s后过界
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



        
            #if len(danger_list) > 0 and not (self.is_safe() and plane['Type'] == 2) and not self.is_regiment_win(plane['RegimentID']): # 逃逸
            if len(danger_list) > 0 and not (self.is_safe() and plane['Type'] == 2): 
                self.escape_toward[plane['ID']] = 0 
                #if len(danger_list) == 1: 
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
            '''test'''
            # print('surround_flag', surround_flag)
            # print('surround_dict', s_dict)
            '''test'''
            # surround_flag = 0

   
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
                    dest = self.rect_dest_bound(dest['X'], dest['Y'], max_bound)
                    leader_fire_route_list = [{"X": dest['X'], "Y": dest['Y'], "Z": data['area_max_alt']-500}, ]
                    speed = 300
                    if plane['Type'] == 1 and not self.is_regiment_win(plane['RegimentID'], True):
                        speed = 270
                    cmd_list.append(CmdEnv.make_linepatrolparam(plane['ID'], leader_fire_route_list,
                                                                speed, data['move_max_acc'],
                                                                data["move_max_g"]))
                else:
                    cmd_list.append(
                        CmdEnv.make_areapatrolparam(plane['ID'], plane['X'],
                                                    plane['Y'], data['area_max_alt'], 100, 100,
                                                    data['move_max_speed'], data['move_max_acc'], data['move_max_g']))

    def layout_plane_select(self, myself): 
        regimentID = myself['RegimentID']
        min_distance = 450000
        select_plane = {}
        for enemy_plane in self.regiments[regimentID]:
            for my_manned in self.my_manned_info:
                distance = TSVector3.distance(enemy_plane, my_manned)
                if distance < min_distance and enemy_plane['LeftWeapon'] > 0:
                    min_distance = distance
                    select_plane = enemy_plane
        if len(select_plane) != 0:
            return select_plane
        for enemy_plane in self.regiments[regimentID]:
            for my_manned in my_manned_info: 
                distance = TSVector3.distance(enemy_plane, my_manned)
                if distance < min_distance:
                    min_distance = distance
                    select_plane = enemy_plane
        return select_plane

    def layout_point_select(self, myself):

        data = self.get_move_data(myself)
        '''
        isblue = 1
        if self.name == 'red':
            isblue = -1
        if myself['Type'] == 1 and self.is_in_center(self.enemy_manned_info[0]) is False:
            return {'X': isblue * 45000 * (1 - myself['CurTime'] / (20 * 60)), 'Y': 0}
        '''
        enemy_leftweapon = 0
        for enemy_plane in self.enemy_allplane_infos:
            enemy_leftweapon += enemy_plane['LeftWeapon']
        for enemy_missile in self.enemy_missile_infos:
            if enemy_missile['EngageTargetID'] == myself['ID']:
                enemy_leftweapon += 1
        if enemy_leftweapon <= 1 and myself['LeftWeapon'] == 1:
            return {'X': random.randint(0,100), 'Y': random.randint(0,100)}



        select_plane = self.layout_plane_select(myself)

        if TSVector3.distance(select_plane, myself) < 0.0001: 
            return {'X': myself['X'] + math.sin(myself['Heading']) * 1000, 'Y': myself['Y'] + math.cos(myself['Heading']) * 1000}

        data0 = self.get_move_data(myself)
        data1 = self.get_move_data(select_plane)

        left_num = 0
        right_num = 0
        my_theta = self.XY2theta(select_plane['X']-myself['X'], select_plane['Y']-myself['Y'])

        if myself['LeftWeapon'] == 0 or myself['Type'] == 1:
            right_num = 0
            left_num = 0
        else:
            for plane in self.my_allplane_infos:
                if TSVector3.distance(plane, select_plane) < 0.0001:
                    continue
                if self.availability(plane) and plane['ID'] != myself['ID'] and plane['RegimentID'] == myself['RegimentID'] and plane['LeftWeapon'] > 2-plane['Type']: #我方载弹飞机包夹敌方
                    plane_theta = self.XY2theta(select_plane['X'] - plane['X'], select_plane['Y'] - plane['Y'])
                    d_theta = self.pi_bound(plane_theta - my_theta)
                    if d_theta >= 0:
                        left_num += 1
                    else:
                        right_num += 1

        my_select_distance = TSVector3.distance(select_plane, myself)
        my_closest_distance = my_select_distance
        closest_plane = {}
        for enemy_plane in self.enemy_allplane_infos:
            if enemy_plane['LeftWeapon'] > 2 - enemy_plane['Type'] and myself['Type'] == 2 or \
                    enemy_plane['LeftWeapon'] > 0 and myself['Type'] == 1:
                distance = TSVector3.distance(enemy_plane, myself)
                if distance < my_closest_distance:
                    my_closest_distance = distance
                    closest_plane = enemy_plane
        if self.safe_flag:
            closest_plane = select_plane

        '''
        if my_closest_distance <= data0['launch_range'] and not self.safe_flag or\
                my_closest_distance <= 20000 and self.safe_flag:
            if myself['LeftWeapon'] == 0 or self.is_regiment_win(myself['RegimentID']) is False or myself['Type'] == 1:
                looking_distance = 58000
                if myself['Type'] == 1:
                    looking_distance = 50000
                elif self.safe_flag:
                    looking_distance = 18000
                return self.get_KeppDistance_move_XY(myself, closest_plane, looking_distance)
            elif closest_plane['ID'] != select_plane['ID']:
                return {'X': closest_plane['X'], 'Y': closest_plane['Y']}
        '''

        if len(closest_plane) != 0 and my_closest_distance <= data0['launch_range'] and not self.safe_flag or self.safe_flag:
            if (myself['LeftWeapon'] == 0 or self.is_regiment_win(myself['RegimentID']) is False or myself['Type'] == 1) and not self.safe_flag:
                looking_distance = 58000
                if myself['Type'] == 1:
                    looking_distance = 50000
                return self.get_KeppDistance_move_XY(myself, closest_plane, looking_distance)
            elif (myself['Type'] == 1 or myself['Type'] == 2 and not self.is_looking(myself, closest_plane, myself['Heading']) and my_closest_distance < 10000) and self.safe_flag:
                looking_distance = 78000
                dest = self.get_KeppDistance_move_XY(myself, closest_plane, looking_distance)
                return self.dest_bound(myself, dest['X'], dest['Y'], 145000, 10000)
            elif closest_plane['ID'] != select_plane['ID']:
                return {'X': closest_plane['X'], 'Y': closest_plane['Y']}




        goal_R = my_select_distance/3+39999
        while my_select_distance <= goal_R:
            goal_R = (goal_R - my_select_distance/3) / 3 + my_select_distance/3
        if left_num != 0 and right_num != 0:
            goal_R /= 2

        turn_theta = math.pi - math.acos(goal_R / my_select_distance)
        dest_theta = my_theta
      
        if left_num < right_num:
            dest_theta = self.pi_bound(dest_theta - turn_theta)
 
        elif right_num < left_num:
            dest_theta = self.pi_bound(dest_theta + turn_theta)

        dest_x = select_plane['X'] + goal_R * math.sin(dest_theta)
        dest_y = select_plane['Y'] + goal_R * math.cos(dest_theta)

        return {'X': dest_x, 'Y': dest_y}

    def get_KeppDistance_move_XY(self, myself, danger, distance):
        bound = 145000
        danger_to_myself_theta = self.XY2theta(myself['X']-danger['X'], myself['Y']-danger['Y'])
        dest_x = danger['X'] + distance * math.sin(danger_to_myself_theta)
        dest_y = danger['Y'] + distance * math.cos(danger_to_myself_theta)
        return self.rect_dest_bound(dest_x, dest_y, bound)

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
        return len(self.my_allplane_infos) < len(self.enemy_allplane_infos) or\
               self.my_manned_info[0]['CurTime'] >= 20 * 60 -300 and self.my_score <= self.enemy_score or\
               self.my_manned_info[1]['CurTime'] >= 20 * 60 -300 and self.my_score <= self.enemy_score


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

            #print(item1['Name'], missile_distance, self.missile_path_distance[120] - self.missile_path_distance[int(start_time)] + 200)

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
                dest['X'] = plane['X'] + dest_distance * math.sin(plane['Heading'])
                dest['Y'] = plane['Y'] + dest_distance * math.cos(plane['Heading'])

       
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

        data = {} # 保存己方机动数据
        if plane['Type'] == 1:  # 所有类型为 1 的飞机是 有人机
            data['move_min_speed'] = 150     # 当前类型飞机的最小速度
            data['move_max_speed'] = 400     # 当前类型飞机的最大速度
            data['move_max_acc'] = 1         # 当前类型飞机的最大加速度
            data['move_max_g'] = 6           # 当前类型飞机的最大超载
            data['area_max_alt'] = 14000     # 当前类型飞机的最大高度
            data['attack_range'] = 1         # 当前类型飞机的最大导弹射程百分比
            data['launch_range'] = 80000     # 当前类型飞机的最大雷达探测范围
        else:                # 所有类型为 2 的飞机是 无人机
            data['move_min_speed'] = 100     # 当前类型飞机的最小速度
            data['move_max_speed'] = 300     # 当前类型飞机的最大速度
            data['move_max_acc'] = 2         # 当前类型飞机的最大加速度
            data['move_max_g'] = 12          # 当前类型飞机的最大超载
            data['area_max_alt'] = 10000     # 当前类型飞机的最大高度
            data['attack_range'] = 1         # 当前类型飞机的最大导弹射程百分比
            data['launch_range'] = 60000     # 当前类型飞机的最大雷达探测范围
        return data # 保存己方机动数据

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
