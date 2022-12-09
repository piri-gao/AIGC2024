# @Author: Yu Zhaoke  
# @Date: 2021-09-27 03:14:04  
# @Last Modified by:   Your name
# @Last Modified time: 2021-10-04 14:13:42

from .net.kz_net import kz_model

def create_rlmodel(batch_size):
    config = {
        'split_list' : [14, 22*5, 20*5, 35*12, 35*12],
        'action_dim' : 6,
        'vnet_outdim': 1,
    }
    data_shape = (None, 5, 1064)
    #data_shape = [feature,]
    model = kz_model(config, debug=False)
    model.set_batchsize(batch_size)  # 设置批大小
    model.build(input_shape= data_shape)  # 编译
    return model