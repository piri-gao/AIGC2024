import py_trees
from agent.bt_agent.nodes.action_base import Action_Base

class Action_Prepare(Action_Base):
    def __init__(self,name="action_prepare"):
        super().__init__(name)
        self.logger_debug("__init__")

    def setup(self):
        super().setup()
        self.logger_debug('setup')

    def update(self):
        self.logger_debug("update")
        self.step_cnt=self.env_client.step_cnt
        self.obs=self.env_client.obs
        return py_trees.common.Status.SUCCESS
