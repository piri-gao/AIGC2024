""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/8/25 10:26
"""

import numpy as np
import copy

def dir2rad(delta_pos):
    result = np.empty(delta_pos.shape[:-1], dtype=complex) #[:-1]就是不要最后一维的数据
    result.real = delta_pos[..., 0]
    result.imag = delta_pos[..., 1]
    rad_angle = np.angle(result)
    return rad_angle

def reg_rad(rad):
    # it's OK to show "RuntimeWarning: invalid value encountered in remainder"
    return (rad + np.pi) % (2 * np.pi) - np.pi

# make angles comparable
def reg_rad_at(rad, ref):
    return reg_rad(rad-ref) + ref

plane = np.array([2, 2])
target = np.array([5, 2])
mates = [np.array([0, 1]), np.array([1, 3])]
n_member = 3
squad_center = plane / n_member
for mate in mates:
    squad_center += mate / n_member

delta = plane - target
delta_all = [mate - target for mate in mates]
delta_all.append(delta)


rad_basement = dir2rad(squad_center - target)

rad_all = [reg_rad_at(dir2rad(D) - rad_basement, 0) for D in delta_all]

rank_index = np.argsort(rad_all)
print(rank_index)
rank_at = np.where(rank_index == (n_member - 1))[0][0]

fan_theta = 110 * (np.pi / 180)
print(rank_at)
uav_theta = rad_basement - fan_theta / 2 + 2 * (fan_theta / (n_member - 1))
print(uav_theta)

vec2d = 1 * np.array([np.cos(uav_theta), np.sin(uav_theta)])

target_3d = np.array([5, 2, 0], dtype=float)
des_now3d = copy.deepcopy(target_3d)
des_now3d[:2] += vec2d
print(des_now3d)

