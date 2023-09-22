# from agent.tink_AI_defence_v0.demo_agent import DemoAgent as BlueAgent
# from agent.tink_AI_studio_nana.demo_agent import DemoAgent as RedAgent
# from agent.tink_AI_Studio.demo_agent import DemoAgent as RedAgent
# from agent.tink_AI_Xs.demo_agent import DemoAgent as BlueAgent
# from agent.QYKZ_final.QYKZ_agent import QYKZ_Agent as BlueAgent
# from agent.QYKZ_v1.QYKZ_agent import QYKZ_Agent as BlueAgent
# from agent.tink_AI_studio_attack0.demo_agent import DemoAgent as RedAgent
# from agent.demo.demo_agent import DemoAgent as BlueAgent
# from agent.WK_AI.wk_agent import WK_Agent as BlueAgent
# from agent.WK_AI_Defence_v2.WK_agent import WK_AI_v2_Agent as BlueAgent
# from agent.WK_AI_Defence.WK_agent import WK_AI_v2_Agent as BlueAgent
from agent.qingkongwanli.demo_agent import DemoAgent as BlueAgent
# from agent.QYKZ_Aggressive.ai_agent import AIAgent as BlueAgent
# from agent.QYKZ_semi.QYKZ_agent import QYKZ_Agent as BlueAgent
# from agent.tink_AI_Xs_MAX.demo_agent import DemoAgent as BlueAgent
# from agent.tink_AI_studio.demo_agent import DemoAgent as BlueAgent
# from agent.QYKZ_final.QYKZ_agent import QYKZ_Agent as RedAgent
# from agent.tink_AI_Studio_attack.demo_agent import DemoAgent as RedAgent
# from agent.tink_AI_v3.demo_agent import DemoAgent as RedAgent
# from agent.tink_AI_v2.demo_agent import DemoAgent as BlueAgent
# from agent.qingkongwanli_show.demo_agent import DemoAgent as RedAgent
# from agent.QMIX_VDN_AIGC.rl_info import DemoAgent as RedAgent
from agent.MAPPO_AIGC.rl_info import DemoAgent as RedAgent
# from agent.tink_AI_defence_v2.demo_agent import DemoAgent as BlueAgent
import random


from agent.demo.demo_agent import DemoAgent as BlueAgent0
from agent.QYKZ_AI.ai_agent import AIAgent as BlueAgent1
from agent.tink_AI.demo_agent import DemoAgent as BlueAgent2
from agent.tink_AI_defence_v1.demo_agent import DemoAgent as BlueAgent3
from agent.tink_AI_defence_v2.demo_agent import DemoAgent as BlueAgent4
from agent.tink_AI_v2.demo_agent import DemoAgent as BlueAgent5
from agent.tink_AI_v3.demo_agent import DemoAgent as BlueAgent6
from agent.tink_AI_v4.demo_agent import DemoAgent as BlueAgent7
from agent.tink_AI_v5.demo_agent import DemoAgent as BlueAgent8
from agent.WK_AI.wk_agent import WK_Agent as BlueAgent9
from agent.WK_AI_Defence.WK_agent import WK_AI_v2_Agent as BlueAgent10
from agent.WK_AI_Defence_v2.WK_agent import WK_AI_v2_Agent as BlueAgent11
from agent.qingkongwanli.demo_agent import DemoAgent as BlueAgent12
# 是否启用host模式,host仅支持单个xsim 
ISHOST = False
# ISHOST = False 
BlueAgents = [globals()["BlueAgent{}".format(i)] for i in range(13)]
# 为态势显示工具域分组ID  1-1000 
HostID = 87
IMAGE = "xsim:v6.5"   # 在xsim:v13.4表演赛基础上授权
# IMAGE = "xsim:v6.3"   # 在xsim:v13.4基础上授权
# IMAGE = "xsim:v6.2"    # 在xsim:v13.2基础上授权
config = {
    "episode_time": 100,   # 训练次数
    "time_ratio": 99, # 引擎倍率, 取值范围[1, 99]
    'agents': {
            'red': RedAgent,
            'blue': BlueAgent
              }
}
config_rl = {
    "episode_time": 100,   # 训练次数
    "time_ratio": 99, # 引擎倍率, 取值范围[1, 99]
    'agents': {
            'red': RedAgent,
            'blue': BlueAgents
              }
}
# 启动XSIM的数量
XSIM_NUM = 5

# 想定名称
scenario_name = "Battle10v10"

ADDRESS = {
    "ip": "127.0.0.1",
    "port": 2022
}
