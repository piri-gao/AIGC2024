import sys
import py_trees


class Action_Base(py_trees.behaviour.Behaviour):
    def __init__(self, name="action_base"):
        super().__init__(name)
        self.logger.debug("%s Action_Base::__init__()" % (self.name))
        self.current_time = None
        self.side_name = None
        self.obs = None
        self.step_cnt = None

    def setup(self):
        self.logger.debug("%s Action_Base::setup()" % (self.name))
        self._init_blackboard()
        self._set_blackboard_keys()

    def _init_blackboard(self):
        self.env_client = py_trees.blackboard.Client(name="env",
                                                     namespace="bt_agent")
        self.action_client = py_trees.blackboard.Client(
            name="action", namespace="bt_agent"
        )
        self.mission_client = py_trees.blackboard.Client(
            name="mission_info", namespace="bt_agent"
        )

    def _set_blackboard_keys(self):
        self.env_client.register_key(key="obs",
                                     access=py_trees.common.Access.READ)
        self.env_client.register_key(key="sim_time",
                                     access=py_trees.common.Access.READ)
        self.action_client.register_key(
            key="action", access=py_trees.common.Access.WRITE
        )
        self.mission_client.register_key(key="side_name",
                                         access=py_trees.common.Access.READ)

    def initialise(self):
        self.logger.debug("%s [Action_Base::initialise()]" % self.name)

    def update(self):
        self.logger.debug("%s [Action_Base::update()]" % self.name)

    def terminate(self, new_status):
        self.logger.debug("%s [Action_Base::treminate()]" % self.name)

    def logger_debug(self, func_name):
        self.logger.debug(
            "%s [%s::%s()]" % (self.name, self.__class__.__name__, func_name)
        )
