"""训练框架配置：dataclass 定义 + yaml 加载。"""
from dataclasses import dataclass, field
from typing import List, Optional, Any

import yaml


@dataclass
class LasVSimCfg:
    endpoint: str = ""
    token: str = ""  # 留空则读环境变量 QX_TOKEN
    local_mode: bool = False  # True 时使用本地 opensim 仿真


@dataclass
class ScenarioCfg:
    task_id: int = 0
    record_id: int = 0


@dataclass
class SimCfg:
    dt: float = 0.1  # 仿真步长(秒)，用于 sim_time 计时与截断判断
    reset_traffic_flow: bool = True  # 每轮 reset 时是否重置交通流


@dataclass
class ActionCfg:
    steer_cmd_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    accel_cmd_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    # 归一化动作 -> 物理量映射
    steer_rad_range: List[float] = field(default_factory=lambda: [-0.6, 0.6])  # 方向盘转角(弧度)
    accel_range: List[float] = field(default_factory=lambda: [-3.0, 3.0])


@dataclass
class ObsCfg:
    v_target: float = 10.0  # 目标车速(m/s)，ego 速度归一化
    w_norm: float = 0.5  # 横摆角速度归一化(rad/s)
    lat_norm: float = 2.0  # 横向偏移归一化(m)
    dist_norm: float = 50.0  # 相对距离归一化(m)
    num_neighbors: int = 5  # 观测中保留的最近车辆数
    ref_points: int = 20  # 参考线采样点数


@dataclass
class DistLatCfg:
    """横向跟踪误差惩罚参数。

    punish_dist_lat = scale * piecewise(t_error) * P_lat
    分段函数（|t_error| 单位 m）：
        < switch_m 时用平方项 t_error^2，误差小时梯度平缓，允许微小居中浮动；
        >= switch_m 时用线性项 linear_slope*|t| + linear_bias，防止换道大误差梯度爆炸。
    """
    scale: float = 5.0
    switch_m: float = 0.3
    linear_slope: float = 0.02
    linear_bias: float = 0.084  # 与平方项在 switch_m 处连续: 0.02*0.3+0.084 = 0.09 = 0.3^2
    lat_discount: float = 1.0  # P_lat 默认值；匝道/汇流场景由状态机经 info["lat_discount"] 覆盖


@dataclass
class RewardCfg:
    weights: dict = field(default_factory=lambda: {
        "speed": 1.0,
        "accel": -0.2,
        "steer": -0.3,
        "collision": -10.0,
        "finish": 20.0,
        "fail": -10.0,
    })
    v_target: float = 10.0  # 目标车速(m/s)，速度奖励基准
    steer_norm: float = 0.6  # 方向盘转角归一化(弧度)
    accel_norm: float = 3.0  # 纵向加速度归一化(m/s^2)
    dist_lat: DistLatCfg = field(default_factory=DistLatCfg)


@dataclass
class TermCfg:
    max_episode_time: float = 60.0  # 单轮最大仿真时长(秒)，超过则 truncated
    collision_terminate: bool = True


@dataclass
class TrainCfg:
    total_timesteps: int = 200_000
    buffer_size: int = 1_000_000  # 经验池(ReplayBuffer)大小
    learning_starts: int = 1_000  # 先积累多少步数据再开始训练
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: Any = "auto"
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    checkpoint_freq: int = 10_000
    log_dir: str = "./logs"
    save_path: str = "./models/sac_lasvsim"


@dataclass
class Cfg:
    lasvsim: LasVSimCfg = field(default_factory=LasVSimCfg)
    scenario: ScenarioCfg = field(default_factory=ScenarioCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    term: TermCfg = field(default_factory=TermCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def _build(cls, data: Optional[dict]):
    obj = cls()
    if data:
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
    return obj


def load_cfg(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Cfg()
    cfg.lasvsim = _build(LasVSimCfg, raw.get("lasvsim"))
    cfg.scenario = _build(ScenarioCfg, raw.get("scenario"))
    cfg.sim = _build(SimCfg, raw.get("sim"))
    cfg.action = _build(ActionCfg, raw.get("action"))
    cfg.obs = _build(ObsCfg, raw.get("obs"))
    cfg.reward = _build(RewardCfg, raw.get("reward"))
    # 嵌套子配置
    if isinstance(cfg.reward.dist_lat, dict):
        cfg.reward.dist_lat = _build(DistLatCfg, cfg.reward.dist_lat)
    cfg.term = _build(TermCfg, raw.get("term"))
    cfg.train = _build(TrainCfg, raw.get("train"))
    return cfg
