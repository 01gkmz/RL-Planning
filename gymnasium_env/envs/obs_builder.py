"""观测构建：将 adapter 的 raw_obs（物理量）加工为定长、归一化的观测向量。"""
from typing import Dict, List

import numpy as np


class ObsBuilder:
    """输出 Dict 观测：

        ego:       [u, v, w, phi, lat_offset] 归一化后的自车状态
        neighbors: N 个最近车辆各 6 维 [dx, dy, cos(dphi), sin(dphi), du, dv]
        route:     K 个参考线采样点各 3 维 [dx, dy, dphi]

    所有坐标均为自车坐标系下的相对量（自车朝前为 x 正方向）。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._n = cfg.num_neighbors
        self._k = cfg.ref_points

    def shapes(self) -> Dict[str, tuple]:
        return {
            "ego": (5,),
            "neighbors": (self._n * 6,),
            "route": (self._k * 3,),
        }

    def build(self, raw_obs: dict) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        ego = self._build_ego(raw_obs)
        neighbors = self._build_neighbors(raw_obs)
        route = self._build_route(raw_obs)
        return {
            "ego": ego.astype(np.float32),
            "neighbors": neighbors.astype(np.float32),
            "route": route.astype(np.float32),
        }

    def _build_ego(self, raw_obs: dict) -> np.ndarray:
        cfg = self.cfg
        u, v, w, u_acc = raw_obs["ego_moving"]
        phi = float(raw_obs["ego_pos"][2])
        lat = float(raw_obs.get("lat_offset", 0.0))
        v_target = max(cfg.v_target, 1e-6)
        return np.array([
            u / v_target,
            v / v_target,
            w / cfg.w_norm,
            phi / np.pi,
            lat / cfg.lat_norm,
        ])

    def _build_neighbors(self, raw_obs: dict) -> np.ndarray:
        cfg = self.cfg
        n, k = self._n, self._k
        ego_xy = raw_obs["ego_pos"][:2]
        ego_phi = float(raw_obs["ego_pos"][2])
        vehicles = raw_obs["vehicles"]  # (M, 7): x, y, phi, u, v, length, width
        if vehicles.shape[0] == 0:
            return np.zeros(n * 6, dtype=np.float32)

        # 自车坐标系旋转矩阵
        c, s = np.cos(ego_phi), np.sin(ego_phi)
        dx = vehicles[:, 0] - ego_xy[0]
        dy = vehicles[:, 1] - ego_xy[1]
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        dphi = vehicles[:, 2] - ego_phi
        dist = np.hypot(dx, dy)
        order = np.argsort(dist)[:n]

        feat = np.zeros(n * 6, dtype=np.float32)
        v_target = max(cfg.v_target, 1e-6)
        for i, idx in enumerate(order):
            dphi_i = dphi[idx]
            feat[i * 6:(i + 1) * 6] = [
                lx[idx] / cfg.dist_norm,
                ly[idx] / cfg.dist_norm,
                np.cos(dphi_i),
                np.sin(dphi_i),
                (vehicles[idx, 3] - raw_obs["ego_moving"][0]) / v_target,
                (vehicles[idx, 4] - raw_obs["ego_moving"][1]) / v_target,
            ]
        return feat

    def _build_route(self, raw_obs: dict) -> np.ndarray:
        cfg = self.cfg
        k = self._k
        line = raw_obs["ref_line"]  # (P, 3): x, y, phi
        ego_xy = raw_obs["ego_pos"][:2]
        ego_phi = float(raw_obs["ego_pos"][2])

        if line.shape[0] == 0:
            return np.zeros(k * 3, dtype=np.float32)

        # 均匀采样 k 个点
        if line.shape[0] >= k:
            idx = np.linspace(0, line.shape[0] - 1, k).astype(int)
        else:
            idx = np.concatenate([
                np.arange(line.shape[0]),
                np.full(k - line.shape[0], line.shape[0] - 1),
            ])
        pts = line[idx]

        c, s = np.cos(ego_phi), np.sin(ego_phi)
        dx = pts[:, 0] - ego_xy[0]
        dy = pts[:, 1] - ego_xy[1]
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        dphi = pts[:, 2] - ego_phi

        feat = np.zeros(k * 3, dtype=np.float32)
        feat[0::3] = lx / cfg.dist_norm
        feat[1::3] = ly / cfg.dist_norm
        feat[2::3] = dphi / np.pi
        return feat
