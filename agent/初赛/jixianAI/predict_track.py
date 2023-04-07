# 对所有情报丢失的情况进行预测

# 预测情报丢失的最长时间
PREDICT_TIME = 6

# 置信度因子
CONFIDENCE_LEVEL_FACTOR = 2
# 衰减因子,一般大于1
ATTENUATION_FACTOR = 1.5

import math

class TrackPredict:
    def __init__(self):
        # 所有情报信息
        self.all_tarck_info = []
        # 预测的情报数据
        self.track_predict_info = []
        # 情报预测的置信度  k:id v:confidence_level
        self.confidence_level_dic = {}

    def save_all_track(self, obs_side):
        for track in obs_side["trackinfos"]:
            for tarck_info in self.all_tarck_info:
                if tarck_info["ID"] == track["ID"]:
                    self.all_tarck_info.remove(tarck_info)
                    self.all_tarck_info.append(track)
                    break

    def track_damage(self, obs_side):
        for missile in obs_side["missileinfos"]:
            for tarck_info in self.all_tarck_info:
                if tarck_info["ID"] == missile["EngageTargetID"]:
                    dis = self._dis_missile_2_plane(missile, tarck_info)
                    if dis < 1400:
                        self.track_predict_info.remove(tarck_info)
                        self.confidence_level_dic.pop(tarck_info["ID"])

    def cal_confidence_level(self, sim_time, track_info):
        interver_time = sim_time - track_info["CurTime"]
        confidence_level = 1 - CONFIDENCE_LEVEL_FACTOR * ((ATTENUATION_FACTOR ** (interver_time-1)) - 1) / (ATTENUATION_FACTOR - 1) / 100
        self.confidence_level_dic[track_info["ID"]] = confidence_level

    def track_predict(self, sim_time, obs_side):
        self.save_all_track(obs_side)
        self.track_predict_info = []
        for track_info in self.all_tarck_info:
            is_track_info_in_obs_side = False
            for obs_side_track in obs_side["trackinfos"]:
                if obs_side_track["ID"] == track_info["ID"]:
                    is_track_info_in_obs_side = True
                    break
            if not is_track_info_in_obs_side and sim_time - track_info["CurTime"] <= PREDICT_TIME:
                self.cal_confidence_level(sim_time, track_info)
                palne_cur_position = self._predict_cur_position(track_info, sim_time)
                track_info["X"] = palne_cur_position[0]
                track_info["Y"] = palne_cur_position[1]
                track_info["Alt"] = palne_cur_position[3]
                track_info["CurTime"] = sim_time
                self.track_predict_info.append(track_info)
        self.track_damage(obs_side)
        return self.track_predict_info, self.confidence_level_dic

    @staticmethod
    def _dis_missile_2_plane(missile, plane):
        dis = math.sqrt(
            (missile["X"] - plane["X"]) ** 2 +
            (missile["Y"] - plane["Y"]) ** 2 +
            (missile["Alt"] - plane["Alt"]) ** 2
        )
        return dis

    @staticmethod
    def _get_vector(heading, pitch):
        heading_radian = math.radians(heading)
        pitch_radian = math.radians(pitch)
        vector_x = - math.sin(heading_radian) * math.cos(pitch_radian)
        vector_y = math.cos(heading_radian) * math.cos(pitch_radian)
        vector_z = math.sin(pitch_radian)
        return [vector_x, vector_y, vector_z]

    def _predict_cur_position(self, origin_palne, sim_time):
        interver_time = sim_time - origin_palne["CurTime"]
        origin_palne_x = origin_palne["X"]
        origin_palne_y = origin_palne["Y"]
        origin_palne_alt = origin_palne["Alt"]
        origin_palne_heading = origin_palne["Heading"]
        origin_palne_pitch = origin_palne["Pitch"]
        origin_palne_speed = origin_palne["Speed"]
        unit_vector = self._get_vector(origin_palne_heading, origin_palne_pitch)
        palne_cur_position = [origin_palne_x + unit_vector[0] * origin_palne_speed * interver_time,
                              origin_palne_y + unit_vector[1] + unit_vector[1] * origin_palne_speed * interver_time,
                              origin_palne_alt + unit_vector[2] * origin_palne_speed * interver_time]
        return palne_cur_position

