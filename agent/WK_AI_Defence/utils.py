""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/8/24 13:49
"""
import numpy as np
from utils.utils_math import TSVector3
from env.env_cmd import CmdEnv

"""计算两架飞机的距离
    args:
        jet1: 1号战机的3d坐标，{"X": X, "Y": Y, "Z": Z}
        jet2: 2号战机的3d租表

    """
def get_dis(jet1, jet2):
    """计算两架飞机的距离

    Args:
        jet1: 1号战机的3d坐标，{"X": X, "Y": Y, "Z": Z}
        jet2: 2号战机的3d租坐标

    Returns:
        距离
    """
    pos1 = jet1.pos3d_dic
    pos2 = jet2.pos3d_dic
    return TSVector3.distance(pos1, pos2)

def minus(jet1, jet2, pos=2):
    """生成的向量是jet2 ->jet1的向量

    Args:
        jet1:
        jet2:
        pos:

    Returns:

    """
    if pos == 2:
        jet1_pos2d = np.array([jet1.X, jet1.Y], dytpe=float)
        jet2_pos2d = np.array([jet2.X, jet2.Y], dytpe=float)
        return jet1_pos2d - jet2_pos2d
    else:
        jet1_pos3d = np.array([jet1.X, jet1.Y, jet1.Z], dytpe=float)
        jet2_pos3d = np.array([jet2.X, jet2.Y, jet1.Z], dytpe=float)
        return jet1_pos3d - jet2_pos3d

def dir2rad(delta_pos):
    result = np.empty(delta_pos.shape[:-1], dtype=complex) #[:-1]就是不要最后一维的数据
    result.real = delta_pos[..., 0]
    result.imag = delta_pos[..., 1]
    rad_angle = np.angle(result)
    return rad_angle


def reg_rad(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def reg_rad_at(rad, ref):
    return (rad - ref + np.pi) % (2 * np.pi) - np.pi + ref

def prob_len(starting_point, direction, dx, lamb, max_try):
    for i in range(max_try):
        dst = starting_point + direction * dx * (i + 1)
        if lamb(dst):
            continue
        else:
            return dx * (i + 1)
    return dx * max_try

def check_and_make_linepatrolparam(host, coord_list, cmd_speed, cmd_accmag, cmd_g):

    def old_way(point):
        if point['X'] > host.MAX_X:
            point['X'] = host.MAX_X  # ; ## print红('if point[X] > host.MAX_X: point[X] = host.MAX_X;')
        if point['X'] < host.MIN_X:
            point['X'] = host.MIN_X  # ; ## print红('if point[X] < host.MIN_X: point[X] = host.MIN_X;')
        if point['Y'] > host.MAX_Y:
            point['Y'] = host.MAX_Y  # ; ## print红('if point[Y] > host.MAX_Y: point[Y] = host.MAX_Y;')
        if point['Y'] < host.MIN_Y:
            point['Y'] = host.MIN_Y  # ; ## print红('if point[Y] < host.MIN_Y: point[Y] = host.MIN_Y;')
        if point['Z'] < host.min_alt:
            point['Z'] = host.min_alt  # ; ### print红('if point[Z] < host.MinHeight: point[Z] = host.MinHeight;')
        if point['Z'] > host.max_alt:
            point['Z'] = host.max_alt  # ; ### print红('if point[Z] > host.MaxHeight: point[Z] = host.MaxHeight;')
        return point

    def avail_coord(point):
        if point['X'] > host.MAX_X:
            return False
        if point['X'] < host.MIN_X:
            return False
        if point['Y'] > host.MAX_Y:
            return False
        if point['Y'] < host.MIN_Y:
            return False
        if point['Z'] < host.min_alt:
            return False
        if point['Z'] > host.max_alt:
            return False
        return True

    def avail_coord_np(point):
        if point[0] > host.MAX_X:
            return False
        if point[0] < host.MIN_X:
            return False
        if point[1] > host.MAX_Y:
            return False
        if point[1] < host.MIN_Y:
            return False
        if point[2] < host.min_alt:
            return False
        if point[2] > host.max_alt:
            return False
        return True

    for i, point in enumerate(coord_list):
        if avail_coord(point):
            continue

        arr = np.array([point["X"], point["Y"], point["Z"]])
        vec_dir_3d = arr - host.pos3d  # 从host指向arr
        vec_dir_3d_unit = vec_dir_3d / (np.linalg.norm(vec_dir_3d) + 1e-7)
        res_len = prob_len(
            starting_point=host.pos3d,
            direction=vec_dir_3d_unit,
            dx=100,
            lamb=avail_coord_np,
            max_try=1000,  # 100km
        )

        res_avail = res_len * vec_dir_3d_unit + host.pos3d
        if res_len < 300:
            # ???? 这种情况比较危险，采用旧的方式
            coord_list[i] = old_way(point)
        else:
            coord_list[i] = old_way({"X": res_avail[0], "Y": res_avail[1], "Z": res_avail[2]})
    return CmdEnv.make_linepatrolparam(host.ID, coord_list, cmd_speed, cmd_accmag, cmd_g)

def adjust_angle_to_target(my_jet, vip, angle):
    rad = angle * np.pi / 180
    delta_oppsite_to_vip = my_jet.pos3d - vip.pos3d
    unit_delta = np.matmul(
        delta_oppsite_to_vip[:2],
        np.array([[np.cos(rad), np.sin(rad)],
                  [np.sin(-rad), np.cos(rad)]]))
    if angle != 0 and angle != 180:
        unit_delta_side2 = np.matmul(
            delta_oppsite_to_vip[:2],
            np.array([[np.cos(rad), np.sin(-rad)],
                      [np.sin(rad), np.cos(rad)]]))

        rad1 = dir2rad(unit_delta)
        rad2 = dir2rad(unit_delta_side2)
        # my_jet_head_rad = np.pi / 2 - my_jet.Heading
        vip_head_rad = np.pi / 2 - vip.Heading
        rad1 = reg_rad_at(rad1, vip_head_rad)
        rad2 = reg_rad_at(rad2, vip_head_rad)
        delta1 = np.abs(rad1 - vip_head_rad) * 180 / np.pi
        delta2 = np.abs(rad2 - vip_head_rad) * 180 / np.pi
        if delta2 > delta1 - 3:  # 另一侧机动
            unit_delta = unit_delta_side2

    H2 = unit_delta[:2] * 100e3 + my_jet.pos2d

    goto_location = [
        {
            "X": H2[0],
            "Y": H2[1],
            "Z": my_jet.Z - 1000  # 因为要下高
        }
    ]
    check_dis(goto_location, my_jet)
    return goto_location

def check_dis(dest, jet):
    d = dest[0]
    dis = np.linalg.norm(jet.pos3d - np.array([d['X'], d['Y'], d['Z']]))
    assert dis > 10e3

def get_angle_deg(p1, p_center, p2):
    vec_p2op_01 = p_center.pos2d - p1.pos2d
    vec_p2op_02 = p_center.pos2d - p2.pos2d
    dir_01 = dir2rad(vec_p2op_01)
    dir_02 = dir2rad(vec_p2op_02)
    dir_02 = reg_rad_at(dir_02, ref=dir_01)
    delta_rad = np.abs(dir_02 - dir_01)  # 弧度制的攻击角度差
    # print("弧度制：", delta_rad)
    delta_deg = delta_rad * 180 / np.pi
    # print("角度制：", delta_deg)
    return delta_deg