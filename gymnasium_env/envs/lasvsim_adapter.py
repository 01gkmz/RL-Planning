"""LasVSim Open API 交互适配层。

将千行仿真平台 (lasvsim) 的 Open API 封装为环境所需的统一接口：
    connect() / reset() / set_control() / step() / end_episode() / close()

数据流：
    step(): 下发控制 -> 平台推演一步 -> 拉取自车/他车状态、参考线、碰撞状态
            -> 打包成 raw_obs（物理量原始数据），由 ObsBuilder 再加工为观测。
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lasvsim_openapi.client import Client
from lasvsim_openapi.http_client import HttpConfig
from lasvsim_openapi.simulator import SimulatorConfig


@dataclass
class AdapterResult:
    """step()/reset() 的统一返回结构。"""
    sim_time: float = 0.0
    raw_obs: Dict[str, Any] = field(default_factory=dict)
    is_running: bool = True
    done_flags: Dict[str, bool] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


class LasVSimAdapter:
    """封装 lasvsim openapi 的 Client / Simulator，向上提供训练环境所需接口。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._client: Optional[Client] = None
        self._sim = None
        self._ego_id: Optional[str] = None
        self._task_id: Optional[int] = cfg.scenario.task_id
        self._record_id: Optional[int] = cfg.scenario.record_id
        self._connected = False
        self._step_count = 0
        self._dt = cfg.sim.dt

    # ---------------- 生命周期 ----------------

    def connect(self, task_id: Optional[int] = None, record_id: Optional[int] = None):
        """建立连接并初始化仿真实例。

        参数为 None 时回退到配置文件中的 scenario.task_id / scenario.record_id，
        避免上层调用 connect() 不传参时把有效值覆盖成 None。
        """
        if self._connected:
            return
        if task_id is None:
            task_id = self.cfg.scenario.task_id
        if record_id is None:
            record_id = self.cfg.scenario.record_id
        self._task_id = task_id
        self._record_id = record_id
        self._new_simulation()
        self._ego_id = self._query_ego_id()
        self._connected = True

    def close(self):
        if self._sim is not None:
            try:
                self._sim.stop()  # 释放平台侧仿真资源
            except Exception:
                pass
        self._sim = None
        self._client = None
        self._connected = False

    def end_episode(self):
        # 不停止仿真器：后续 reset() 会复用同一仿真实例(reset 接口)
        pass

    def _new_simulation(self):
        """从平台拷贝剧本并初始化一个仿真实例。"""
        self._client = self._init_client()
        print(f"DEBUG: task_id={self._task_id!r}, record_id={self._record_id!r}")
        new_record = self._client.process_task.copy_record(self._task_id, self._record_id)
        self._sim = self._client.init_simulator_from_config(
            SimulatorConfig(
                scen_id=new_record.scen_id,
                scen_ver=new_record.scen_ver,
                sim_record_id=new_record.sim_record_id,
            )
        )

    def _init_client(self):
        if self.cfg.lasvsim.local_mode:
            return Client(local_mode=True)
        endpoint = self.cfg.lasvsim.endpoint or os.getenv("QX_ENDPOINT", "")
        token = self.cfg.lasvsim.token or os.getenv("QX_TOKEN", "")
        if not endpoint or not token:
            raise ValueError(
                "缺少 lasvsim 连接配置：请在配置文件中填写 endpoint/token，"
                "或设置环境变量 QX_ENDPOINT / QX_TOKEN"
            )
        return Client(HttpConfig(endpoint=endpoint, token=token))

    # ---------------- 环境接口 ----------------

    def reset(
        self,
        task_id: Optional[int] = None,
        record_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> AdapterResult:
        """重置仿真实例。

        task_id / record_id 为 None 时从配置读取，兼容 Gymnasium 的 reset(seed=...) 调用。
        """
        if task_id is None:
            task_id = self.cfg.scenario.task_id
        if record_id is None:
            record_id = self.cfg.scenario.record_id

        if not self._connected:
            self.connect(task_id, record_id)
        elif (task_id, record_id) != (self._task_id, self._record_id):
            # 场景切换：重建仿真
            self.close()
            self.connect(task_id, record_id)
        else:
            try:
                self._sim.reset(reset_traffic_flow=self.cfg.sim.reset_traffic_flow)
            except Exception:
                # reset 失败则重建仿真实例兜底
                self.close()
                self.connect(task_id, record_id)
            try:
                self._ego_id = self._query_ego_id()
            except Exception:
                pass

        self._step_count = 0
        return self._collect(self._sim)

    def set_control(self, steer_deg: float, lon_acc: float):
        self._sim.set_vehicle_control_info(self._ego_id, steer_deg, lon_acc)

    def step(self) -> AdapterResult:
        step_res = self._sim.step()
        self._step_count += 1
        return self._collect(self._sim, step_res)

    # ---------------- 数据采集 ----------------

    def _query_ego_id(self) -> str:
        test_vehicles = self._sim.get_test_vehicle_id_list()
        if not test_vehicles.list:
            raise RuntimeError("场景中没有测试车辆（test vehicle）")
        return test_vehicles.list[0]

    def _collect(self, sim, step_res=None) -> AdapterResult:
        code = getattr(step_res, "code", None)
        message = getattr(step_res, "message", "") if step_res else ""
        is_running = code.is_running() if code is not None else True
        code_val = int(code.value) if code is not None else 0

        # 拉取状态；单接口失败不影响整体（用默认值兜底）
        ego_pos = self._safe(self._get_positions, [self._ego_id])
        ego_moving = self._safe(self._get_moving, [self._ego_id])
        ego_control = self._safe(self._get_control, [self._ego_id])
        collision = bool(self._safe(self._get_collision, self._ego_id))
        ref_line = self._safe(self._get_ref_line, self._ego_id)
        other_vehicles = self._safe(self._get_other_vehicles)

        ep = ego_pos.get(self._ego_id) if ego_pos else {}
        em = ego_moving.get(self._ego_id) if ego_moving else {}
        ec = ego_control.get(self._ego_id) if ego_control else {}

        x = ep.get("x", 0.0)
        y = ep.get("y", 0.0)
        phi = ep.get("phi", 0.0)
        lat_offset = ep.get("lat_offset", 0.0)
        u = em.get("u", 0.0)
        v = em.get("v", 0.0)
        w = em.get("w", 0.0)
        u_acc = em.get("u_acc", 0.0)
        steer = ec.get("ste_wheel", 0.0)
        lon_acc = ec.get("lon_acc", 0.0)

        finished = code is not None and not is_running and code.value == 1001
        failed = code is not None and not is_running and code.value == 1002

        raw_obs = {
            "ego_pos": np.array([x, y, phi], dtype=np.float32),
            "ego_moving": np.array([u, v, w, u_acc], dtype=np.float32),
            "lat_offset": float(lat_offset),
            "vehicles": other_vehicles if other_vehicles is not None else np.zeros((0, 7), dtype=np.float32),
            "ref_line": ref_line if ref_line is not None else np.zeros((0, 3), dtype=np.float32),
            "collision": collision,
        }

        done_flags = {"collision": collision, "finished": finished, "failed": failed}

        info = {
            "step_code": code_val,
            "step_message": message,
            "ego_state": {
                "u": u, "v": v, "w": w, "phi": phi,
                "lat_offset": lat_offset, "u_acc": u_acc,
                "steer": steer, "lon_acc": lon_acc,
            },
            "collision": collision,
            "finished": finished,
            "failed": failed,
            "n_vehicles": int(len(raw_obs["vehicles"])),
        }

        return AdapterResult(
            sim_time=self._step_count * self._dt,
            raw_obs=raw_obs,
            is_running=is_running,
            done_flags=done_flags,
            info=info,
        )

    # ---------------- 平台数据拉取（异常兜底） ----------------

    @staticmethod
    def _safe(fn, *args, default=None):
        try:
            return fn(*args)
        except Exception:
            return default

    def _get_positions(self, id_list: List[str]) -> Dict[str, Dict[str, float]]:
        res = self._sim.get_vehicle_position(id_list)
        out = {}
        for vid, pos in res.position_dict.items():
            p = pos.point
            out[vid] = {
                "x": float(p.x) if p else 0.0,
                "y": float(p.y) if p else 0.0,
                "phi": float(pos.phi or 0.0),
                "lat_offset": float(pos.t if pos.t is not None else (pos.lane_offset or 0.0)),
            }
        return out

    def _get_moving(self, id_list: List[str]) -> Dict[str, Dict[str, float]]:
        res = self._sim.get_vehicle_moving_info(id_list)
        return {
            vid: {
                "u": float(m.u or 0.0),
                "v": float(m.v or 0.0),
                "w": float(m.w or 0.0),
                "u_acc": float(m.u_acc or 0.0),
            }
            for vid, m in res.moving_info_dict.items()
        }

    def _get_control(self, id_list: List[str]) -> Dict[str, Dict[str, float]]:
        res = self._sim.get_vehicle_control_info(id_list)
        return {
            vid: {"ste_wheel": float(c.ste_wheel or 0.0), "lon_acc": float(c.lon_acc or 0.0)}
            for vid, c in res.control_info_dict.items()
        }

    def _get_collision(self, vehicle_id: str) -> bool:
        res = self._sim.get_vehicle_collision_status(vehicle_id)
        return bool(res.collision_status)

    def _get_ref_line(self, vehicle_id: str) -> np.ndarray:
        res = self._sim.get_vehicle_reference_lines(vehicle_id)
        if not res.reference_lines:
            return np.zeros((0, 3), dtype=np.float32)
        line = max(res.reference_lines, key=lambda l: len(l.points))
        pts = [(p.x, p.y, 0.0) for p in line.points[:200]]
        return np.asarray(pts, dtype=np.float32)

    def _get_other_vehicles(self) -> np.ndarray:
        all_ids = self._sim.get_vehicle_id_list()
        ids = [vid for vid in all_ids.list if vid != self._ego_id]
        if not ids:
            return np.zeros((0, 7), dtype=np.float32)

        positions = self._get_positions(ids)
        movings = self._get_moving(ids)
        base = {}
        try:
            res = self._sim.get_vehicle_base_info(ids)
            base = {
                vid: {
                    "length": float(info.base_info.length or 0.0) if info.base_info else 0.0,
                    "width": float(info.base_info.width or 0.0) if info.base_info else 0.0,
                }
                for vid, info in res.info_dict.items()
            }
        except Exception:
            pass

        rows = []
        for vid in ids:
            p = positions.get(vid, {})
            m = movings.get(vid, {})
            b = base.get(vid, {})
            rows.append([
                p.get("x", 0.0), p.get("y", 0.0), p.get("phi", 0.0),
                m.get("u", 0.0), m.get("v", 0.0),
                b.get("length", 4.8), b.get("width", 1.9),
            ])
        return np.asarray(rows, dtype=np.float32)