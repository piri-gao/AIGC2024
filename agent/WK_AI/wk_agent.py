from typing import List

from agent.agent import Agent
from agent.WK_AI.observation import GlobalObservation
from agent.WK_AI.decision_making import WK_Decision

class WK_Agent(Agent):
    """
        自定义智能体
    """
    def __init__(self, name, config):
        super(WK_Agent, self).__init__(name, config['side'])
        self._init()

    def _init(self):
        # 全局态势
        self.global_observation = GlobalObservation()
        # 指挥决策
        self.commond_decision = WK_Decision(self.global_observation)

    def step(self, sim_time, obs, **kwargs) -> List[dict]:

        cmd_list = []
        self.global_observation.update_observation(obs)
        self.commond_decision.update_decision(sim_time, cmd_list)
        return cmd_list