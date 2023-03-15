from typing import List
from agent.agent import Agent
from agent.QYKZ_semi.observation import GlobalObservation
from agent.QYKZ_semi.decision_making import QYKZ_Decision

class QYKZ_Agent(Agent):
    """
        自定义智能体
    """
    def __init__(self, name, config):
        super(QYKZ_Agent, self).__init__(name, config['side'])
        self._init()

    def _init(self):
        # 全局态势
        self.global_observation = GlobalObservation()
        # 指挥决策
        self.commond_decision = QYKZ_Decision(self.global_observation)

    def reset(self):
        self.commond_decision.reset()

    def step(self, sim_time, obs, **kwargs) -> List[dict]:
        self.global_observation.update_observation(obs)
        cmd_list = self.commond_decision.update_decision(sim_time)
        return cmd_list