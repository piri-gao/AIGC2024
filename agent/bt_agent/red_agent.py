from typing import List
import py_trees
from agent.bt_agent.parser import create_tree_from_json

from agent.agent import Agent

class RedAgent(Agent):
    """
        自定义智能体
    """
    def __init__(self, name, config):
        super().__init__(name, config['side'])
        self._init()

    def _init(self):
        self.root=None

        self.setup()

    def step(self, sim_time, obs, **kwargs) -> List[dict]:
        self.action_client.action=[]
        self.env_client.obs=obs
        self.env_client.sim_time=sim_time
        self.root.tick_once()
        cmd_list = self.action_client.action
        return cmd_list


    def setup(self, json_file='agent/bt_agent/red.json', side_name="red"):
        self._init_blackboard()
        self._set_blackboard_keys()
        self.mission_client.side_name = side_name

        self.root = create_tree_from_json(json_file)
        self.root.setup_with_descendants()

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
                                     access=py_trees.common.Access.WRITE)
        self.env_client.register_key(key="sim_time",
                                     access=py_trees.common.Access.WRITE)
        self.action_client.register_key(
            key="action", access=py_trees.common.Access.WRITE
        )
        self.mission_client.register_key(key="side_name",
                                         access=py_trees.common.Access.WRITE)


