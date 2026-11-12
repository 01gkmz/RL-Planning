"""动作映射：归一化动作 [-1,1]^2 -> 物理控制量 (方向盘转角°, 纵向加速度)。"""
from typing import Tuple

import numpy as np


class ActionMapper:
    def __init__(self, cfg):
        self._steer_cmd = np.asarray(cfg.steer_cmd_range, dtype=np.float32)
        self._accel_cmd = np.asarray(cfg.accel_cmd_range, dtype=np.float32)
        self._steer_deg = np.asarray(cfg.steer_deg_range, dtype=np.float32)
        self._accel = np.asarray(cfg.accel_range, dtype=np.float32)

    @staticmethod
    def _map(x: float, src: np.ndarray, dst: np.ndarray) -> float:
        ratio = (x - src[0]) / max(src[1] - src[0], 1e-9)
        return float(dst[0] + ratio * (dst[1] - dst[0]))

    def to_physical(self, action: np.ndarray) -> Tuple[float, float]:
        action = np.asarray(action, dtype=np.float32)
        steer_deg = self._map(action[0], self._steer_cmd, self._steer_deg)
        lon_acc = self._map(action[1], self._accel_cmd, self._accel)
        return steer_deg, lon_acc
