import py_trees
from agent.bt_agent.nodes.action_base import Action_Base

class Demo_Action2(Action_Base):
    def __init__(self,name="demo_action2"):
        super().__init__(name)
        self.logger_debug("__init__")

    def setup(self):
        super().setup()
        self.logger_debug('setup')

    def update(self):
        self.logger_debug("update")
        self.obs=self.env_client.obs
        return py_trees.common.Status.SUCCESS
