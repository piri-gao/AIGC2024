from typing import List
from agent.agent import Agent
from agent.WK_AI_Defence_v2.observation import GlobalObservation
from agent.WK_AI_Defence_v2.decision_making import WK_v2_Decision

class WK_AI_v2_Agent(Agent):
    """
        自定义智能体
    """
    def __init__(self, name, config):
        super(WK_AI_v2_Agent, self).__init__(name, config['side'])
        self._init()

    def _init(self):
        # 全局态势
        self.global_observation = GlobalObservation()
        # 指挥决策
        self.commond_decision = WK_v2_Decision(self.global_observation)

    def reset(self):
        self.commond_decision.reset()

    def step(self, sim_time, obs, **kwargs) -> List[dict]:
        self.global_observation.update_observation(obs)
        cmd_list = self.commond_decision.update_decision(sim_time)
        return cmd_list