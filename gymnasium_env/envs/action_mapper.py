"""动作映射：归一化动作 [-1,1]^2 -> 物理控制量 (方向盘转角弧度, 纵向加速度)。

注意：lasvsim 平台的 ste_wheel 按弧度解释（SDK 未注明单位，
但平台实测确认；发送角度值会导致转向饱和）。
"""
from typing import Tuple

import numpy as np


class ActionMapper:
    def __init__(self, cfg):
        self._steer_cmd = np.asarray(cfg.steer_cmd_range, dtype=np.float32)
        self._accel_cmd = np.asarray(cfg.accel_cmd_range, dtype=np.float32)
        self._steer_rad = np.asarray(cfg.steer_rad_range, dtype=np.float32)
        self._accel = np.asarray(cfg.accel_range, dtype=np.float32)

    @staticmethod
    def _map(x: float, src: np.ndarray, dst: np.ndarray) -> float:
        ratio = (x - src[0]) / max(src[1] - src[0], 1e-9)
        return float(dst[0] + ratio * (dst[1] - dst[0]))

    def to_physical(self, action: np.ndarray) -> Tuple[float, float]:
        action = np.asarray(action, dtype=np.float32)
        steer_rad = self._map(action[0], self._steer_cmd, self._steer_rad)
        lon_acc = self._map(action[1], self._accel_cmd, self._accel)
        return steer_rad, lon_acc
