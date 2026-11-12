"""奖励函数：基于自车物理状态、动作与终止标志计算标量奖励。

各部分（权重可在配置中调整）：
    speed     跟车速度奖励: min(u, v_target) / v_target
    lateral   横向偏移惩罚: -|lat_offset|
    accel     纵向加速度惩罚: -|lon_acc| / accel_norm
    steer     方向盘转角惩罚: -(steer / steer_norm)^2
    collision 碰撞惩罚
    finish    正常跑完场景奖励 / fail 失败惩罚
"""
from typing import Dict

import numpy as np


class RewardFn:
    def __init__(self, cfg):
        self._w = cfg.weights
        self._v_target = max(float(cfg.v_target), 1e-6)
        self._steer_norm = max(float(cfg.steer_norm), 1e-6)
        self._accel_norm = max(float(cfg.accel_norm), 1e-6)

    def compute(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        prev_obs: Dict[str, np.ndarray],
        done_flags: Dict[str, bool],
        info: Dict,
    ) -> float:
        ego = info.get("ego_state", {}) if isinstance(info, dict) else {}
        u = float(ego.get("u", 0.0))
        lat = float(ego.get("lat_offset", 0.0))
        steer = float(ego.get("steer", 0.0))
        lon_acc = float(ego.get("lon_acc", 0.0))

        r = 0.0
        r += self._w.get("speed", 0.0) * min(u, self._v_target) / self._v_target
        r += self._w.get("lateral", 0.0) * abs(lat)
        r += self._w.get("accel", 0.0) * abs(lon_acc) / self._accel_norm
        r += self._w.get("steer", 0.0) * (steer / self._steer_norm) ** 2

        if done_flags.get("collision"):
            r += self._w.get("collision", 0.0)
        if done_flags.get("finished"):
            r += self._w.get("finish", 0.0)
        if done_flags.get("failed"):
            r += self._w.get("fail", 0.0)
        return float(r)
