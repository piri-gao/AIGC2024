import py_trees
from agent.bt_agent.nodes.action_base import Action_Base
class If_In_Time(Action_Base):
    def __init__(self, end_time=0,start_time=0,name='If_Tn_Time'):
        super().__init__(name)
        self.end_time = end_time
        self.start_time = start_time
    def setup(self):
        super().setup()
        self.logger.debug("%s [If_In_Time::setup()]"%self.name)
        # self.name=self.parent.name+" "+self.name
    def update(self):
        # 每次都会调用的最核心函数.
        self.logger.debug("%s [If_Tn_Time::update()]"%self.name)
        self.obs=self.env_client.obs
        if self.end_time:
            if self.obs['current_time']< self.end_time:
                if self.start_time:
                    if self.obs['current_time'] > self.start_time:
                        return py_trees.common.Status.SUCCESS
                    else:
                        return py_trees.common.Status.FAILURE
                else:
                    return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        else:
            if self.start_time:
                if self.obs['current_time'] > self.start_time:
                    return py_trees.common.Status.SUCCESS
                else:
                    return py_trees.common.Status.FAILURE
            else:
                return py_trees.common.Status.SUCCESS

