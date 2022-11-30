from agent.demo.demo_agent import DemoAgent as RedAgent
from agent.demo.demo_agent import DemoAgent as BlueAgent
#from agent.jixianAI.jixian_agent import JixianXsimAIAgent as RedAgent


# 是否启用host模式,host仅支持单个xsim
ISHOST = True
# ISHOST = False

# 为态势显示工具域分组ID  1-1000
HostID = 71

# IMAGE = "xsim:v10.0"
IMAGE = "xsim:v14.0"    # 在xsim:v10.0基础上授权

config = {
    "episode_time": 10,   # 训练次数
    "time_ratio": 99, # 引擎倍率, 取值范围[1, 99]
    'agents': {
            'red': RedAgent,
            'blue': BlueAgent
              }
}

# 启动XSIM的数量
XSIM_NUM = 3

# 想定名称
scenario_name = "Battle10v10"

ADDRESS = {
    "ip": "192.168.21.129",
    "port": 2022
}
