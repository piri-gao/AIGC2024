""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/8/9 12:54
"""
from typing import List
from agent.agent import Agent
#from agent.jixianAI_demo.observation import GlobalObservation
from agent.jixianAI.observation import GlobalObservation
#from agent.jixianAI_demo.decision_making import JixianDecision
#from agent.jixianAI.decision_making import JixianDecision

"""
战术思路：
不基于AIdemo

获取信息:
更新observation
--------------------------------

进攻策略：进攻策略内容复杂，写作了一个单独的类来调用更新
0）接敌策略：战术移动策略
1）威胁排序
2）开火策略，已经开火后的跟随策略
3）制导策略

tip记录:
编队任务是否初始化后就不变了
--------------------------------------
防御：
躲避策略，需要重新写，算的更精细一些：和导弹有更大的角度差，上下角度改变是为了和导弹有更大的距离差
躲避后需要回来

编写不少于4种躲避策略，以最少应对4种情况

tip记录：
1.如果我方某架飞机无弹，那么它需要执行特定的策略，例如只制导不躲避
-------------------------------------------------
整体思路：
1. 需要加入10v10特有的策略
2. 是否要设置不同任务的优先级，还是设计一套逻辑判断当前态势下哪个任务最重要
3. 判断当前局势，不同的局势执行不同的策略：使用聚类算法:
    待考虑：
    1. 
    # 聚类算法特征：敌人的数量，敌人弹药数，在飞导弹数，我方剩余飞机数量，我方导弹数量，和敌人的距离
    # 对敌人分布进行聚类分析：
    #     聚集
    # 根据场上局势不同，分别选择：
    #     优势战术：
    #     以多打少战术
    #     势均力敌战术：初始就是势均力敌的情况
    #     劣势：保护我方有人机
    #     大劣势：猛攻敌人有人机
    2. 在一个地方用到强化学习算法，哪里还未知
    
    
    3. 已实现：对敌人的位置分布进行聚类分析：

-------------------------------------------------
战术移动策略：
1.初始化时，朝中心点移动
2.接敌时一定距离以小队包夹战术靠近敌人
3.制导以小队编队形式制导
---------------------------------------------------
实体：
1.为我方飞机加入状态，飞机分为持续状态动作和单步长状态动作
持续状态如一些躲避动作，持续状态在满足一些触发条件后，会结束持续状态改为默认状态重新更新
持续状态：
实时状态：
    如果是攻击状态：必须朝向敌人，开火，这一过程完成前不可以打断
    如果是有人机逃逸状态：则必须到达安全距离，才可以终止此状态

--------------------------------------------
情报预测：class
如果被干扰：干扰范围100km，被干扰的飞机都没有探测能力
    1.判断是否被干扰，
    2.如果被干扰，做位置预判，trackinfo，
    躲避 和 导弹
    敌机 
    消失的三个原因：
        是否阵亡：瞄准的导弹是否
        是否飞出探测距离：80公里
        是否被干扰：消失了，3-6秒，拿到6的秒的数据
------------------------------------    
干扰动作：
    是否可执行干扰：硬性条件
    该不该执行干扰：飞机的状态
    for plane in self.my_plane:
        敌方飞机（num）在我的探测范围里，并且它没发弹
        plane.status = "ganrao"：
            cmd.list(ganrao)
        
        攻击  
        fol
-----------------------------------
测试

有人机，调用几架飞机，初始化位置，移动（保证可以相遇），开火（探测到敌人后会开火），躲避（探测到导弹来袭回180度调头往后跑）
可以放我自己的策略
-------------------
"""
from typing import List
from agent.agent import Agent
from agent.jixianAI.observation import GlobalObservation
from agent.jixianAI.decision_making import JixianXsimAI

class JixianXsimAIAgent(Agent):
    """
        自定义智能体
    """
    def __init__(self, name, config):
        super(JixianXsimAIAgent, self).__init__(name, config['side'])
        self._init()

    def _init(self):
        # 全局态势
        self.global_observation = GlobalObservation()
        # 指挥决策
        self.commond_decision = JixianXsimAI(self.global_observation)

    def reset(self):
        self.commond_decision.reset()

    def step(self, sim_time, obs, **kwargs) -> List[dict]:
        self.global_observation.update_observation(obs)
        cmd_list = self.commond_decision.update_decision(sim_time)
        return cmd_list