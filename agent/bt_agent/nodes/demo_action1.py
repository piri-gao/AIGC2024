import py_trees
from agent.bt_agent.nodes.action_base import Action_Base
from agent.demo.observation import GlobalObservation
from agent.demo.decision_making import DemoDecision

class Demo_Action1(Action_Base):
    def __init__(self,name="demo_action1"):
        super().__init__(name)
        self.logger_debug("__init__")

    def setup(self):
        super().setup()
        self.logger_debug('setup')
        # 全局态势
        self.global_observation = GlobalObservation()
        # 指挥决策
        self.commond_decision = DemoDecision(self.global_observation)

    def update(self):
        self.logger_debug("update")
        cmd_list=[]
        obs=self.env_client.obs
        sim_time=self.env_client.sim_time
        self.global_observation.update_observation(obs)
        self.commond_decision.update_decision(sim_time, cmd_list)
        self.action_client.action+=cmd_list
        return py_trees.common.Status.SUCCESS
