"""奖励函数：基于自车物理状态、动作与终止标志计算标量奖励。

各部分（权重可在配置中调整）：
    speed     跟车速度奖励: min(u, v_target) / v_target
    dist_lat  横向跟踪误差惩罚（分段二次/线性，见 DistLatCfg）:
              t_error = -(ego_x-ref_x)*sin(ref_phi) + (ego_y-ref_y)*cos(ref_phi)
              punish_dist_lat = 5.0 * piecewise(t_error) * P_lat
    accel     纵向加速度惩罚: -|lon_acc| / accel_norm
    steer     方向盘转角惩罚: -(steer / steer_norm)^2
    collision 碰撞惩罚
    finish    正常跑完场景奖励 / fail 失败惩罚

P_lat：默认 1.0。匝道/主路汇流等场景下，状态机可通过 info["lat_discount"]
对目标车道注入折扣（0~1），降低目标车道的横向惩罚，鼓励换道决策。
"""
from typing import Dict

import numpy as np


class RewardFn:
    def __init__(self, cfg):
        self._w = cfg.weights
        self._v_target = max(float(cfg.v_target), 1e-6)
        self._steer_norm = max(float(cfg.steer_norm), 1e-6)
        self._accel_norm = max(float(cfg.accel_norm), 1e-6)

        dl = cfg.dist_lat
        self._lat_scale = float(dl.scale)
        self._lat_switch = float(dl.switch_m)
        self._lat_slope = float(dl.linear_slope)
        self._lat_bias = float(dl.linear_bias)
        self._lat_discount = float(dl.lat_discount)

    def _dist_lat_base(self, t_error: float) -> float:
        """分段函数: |t|<0.3m 平方项(梯度平缓)，|t|>=0.3m 线性项(防梯度爆炸)。

        两段在 |t|=0.3 处连续: 0.3^2 = 0.02*0.3 + 0.084 = 0.09。
        """
        t = abs(t_error)
        if t < self._lat_switch:
            return t * t
        return self._lat_slope * t + self._lat_bias

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
        t_error = float(ego.get("t_error", 0.0))
        steer = float(ego.get("steer", 0.0))
        lon_acc = float(ego.get("lon_acc", 0.0))

        # P_lat：优先取状态机注入的目标车道折扣，否则用配置默认值
        if isinstance(info, dict) and "lat_discount" in info:
            p_lat = float(info["lat_discount"])
        else:
            p_lat = self._lat_discount

        r = 0.0
        r += self._w.get("speed", 0.0) * min(u, self._v_target) / self._v_target
        r -= self._lat_scale * self._dist_lat_base(t_error) * p_lat
        r += self._w.get("accel", 0.0) * abs(lon_acc) / self._accel_norm
        r += self._w.get("steer", 0.0) * (steer / self._steer_norm) ** 2

        if done_flags.get("collision"):
            r += self._w.get("collision", 0.0)
        if done_flags.get("finished"):
            r += self._w.get("finish", 0.0)
        if done_flags.get("failed"):
            r += self._w.get("fail", 0.0)
        return float(r)
