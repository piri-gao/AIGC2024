import json
from py_trees.decorators import OneShot
from py_trees.composites import Sequence,Selector,Parallel

from agent.bt_agent.nodes.rl_node import RLNode
from agent.bt_agent.nodes.if_in_time import If_In_Time

from agent.bt_agent.nodes.demo_action1 import Demo_Action1
from agent.bt_agent.nodes.demo_action2 import Demo_Action2
str2btnode=dict(
    Parallel = Parallel,
    Sequence = Sequence,
    Selector = Selector,
    IfExecuted=OneShot,
    RL_Agent=RLNode,
    If_In_Time=If_In_Time,
    demo_action1=Demo_Action1,
    demo_action2=Demo_Action2
)
def create_tree_from_dict(tree_dict):
    root_cls_name=tree_dict['type']
    root_cls=str2btnode[root_cls_name]
    params_dict={}
    if "properties" in tree_dict:
        for dic in tree_dict["properties"]:
            params_dict[dic['name']]=dic['value']
    params_tmp={}
    for k,v in params_dict.items():
        if v.startswith('[') or v.startswith('{') or v[0] in ('0','1','2','3','4','5','6','7','8','9'):
            params_tmp[k]=eval(v)
        else:
            params_tmp[k]=v
    root=root_cls(**params_tmp)
    if "label" in tree_dict:
        root.label=tree_dict['label']
    decorator_node_cls=[]
    if "decorators" in tree_dict:
        decor_list=tree_dict["decorators"]
        for decor_dict in decor_list:
            decor_cls_name=decor_dict['type']
            decor_cls=str2btnode[decor_cls_name]
            decorator_node_cls.append(decor_cls)
    child_nodes=[]
    if "childNodes" in tree_dict:
        for child_dict in tree_dict['childNodes']:
            child_node=create_tree_from_dict(child_dict)
            child_nodes.append(child_node)
    if len(child_nodes)>0:
        root.add_children(child_nodes)
    result=root
    for decor_cls in reversed(decorator_node_cls):
        decor_obj=decor_cls(child=result)
        result=decor_obj
    return result
def create_tree_from_json(json_path):
    root_info=json.load(open(json_path))
    root=create_tree_from_dict(root_info)
    return root

if __name__=="__main__":
    root=create_tree_from_json('red.json')