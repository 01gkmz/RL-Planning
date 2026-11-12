# rl-planning：基于 SB3 SAC + LasVSim 的强化学习训练框架

在千行仿真平台（LasVSim）上，用 stable-baselines3 的 SAC 算法训练自动驾驶决策/控制策略。框架通过 LasVSim Open API 与仿真平台交互采集数据，存入 SAC 内置经验池（ReplayBuffer），积累一定步数后开始训练。

## 数据流总览

```
train.py (SB3 SAC)
   └─ LasVSimEnv (gymnasium 环境)
        ├─ LasVSimAdapter ── lasvsim-openapi ── 千行仿真平台
        │    (copy_record / step / reset / 拉取自车·他车·参考线·碰撞状态)
        ├─ ObsBuilder      原始数据 → Dict 观测 (ego / neighbors / route)
        ├─ ActionMapper    归一化动作 → 方向盘转角(°) / 纵向加速度
        ├─ RewardFn        速度跟踪 + 平滑性惩罚 + 碰撞/完成事件
        └─ compute_termination  仿真结束/碰撞 → terminated，超时 → truncated
```

训练循环：每步交互产生 `(s, a, r, s', done)` 存入经验池；经验池攒满 `learning_starts` 步后，SAC 开始采样训练；之后边收集边训练，直至 `total_timesteps`。

## 目录结构

```
rl-planning/
├── train.py                      # 训练入口
├── requirements.txt              # 依赖
├── configs/
│   └── sac_lasvsim.yaml          # 全部训练/环境参数
└── gymnasium_env/
    ├── __init__.py               # 注册 LasVSim-v0
    ├── config.py                 # 配置 dataclass + yaml 加载
    └── envs/
        ├── lasvsim_env.py        # gymnasium 环境主体
        ├── lasvsim_adapter.py    # LasVSim Open API 交互层（平台数据采集）
        ├── obs_builder.py        # 观测构建
        ├── action_mapper.py      # 动作映射
        ├── reward.py             # 奖励函数
        └── termination.py        # 终止判断
```

## 安装

```bash
pip install -r requirements.txt
```

依赖：`stable-baselines3`、`gymnasium`、`torch`、`numpy`、`pyyaml`、`lasvsim-openapi`。

## 配置

所有参数在 [configs/sac_lasvsim.yaml](configs/sac_lasvsim.yaml)，关键项：

### 平台连接（lasvsim 段）

```yaml
lasvsim:
  endpoint: "https://qianxing-api.risenlighten.com"  # 或环境变量 QX_ENDPOINT
  token: ""                                          # 留空则读环境变量 QX_TOKEN
  local_mode: false                                  # true 使用本地 opensim 仿真
```

- `token`：登录千行平台后，在个人中心页面底部复制；也可设置环境变量 `QX_TOKEN`
- `local_mode`：无平台账号时可先用本地 opensim 调试

### 场景（scenario 段）

```yaml
scenario:
  task_id: 0     # 平台任务ID：任务页 URL 中的 id= 参数
  record_id: 0   # 平台剧本ID：剧本页 URL 中的 record_id= 参数
```

两项必须填真实值（为 0 时启动会打印警告）。

### 动作与观测（action / obs 段）

- `action.steer_rad_range` / `accel_range`：归一化动作映射到的物理量程。**转向角平台按弧度解释**（实测确认，勿填角度值），需按平台车型实际控制范围调整
- `obs.num_neighbors` / `ref_points`：观测中保留的最近车辆数、参考线采样点数

### 奖励（reward 段）

奖励 = 各子项加权和，详见 [reward.py](gymnasium_env/envs/reward.py)：

```yaml
reward:
  v_target: 10.0      # 速度奖励目标(m/s)
  weights:
    speed: 1.0        # min(u, v_target)/v_target
    lateral: -0.5     # -|横向偏移|
    accel: -0.2       # 加速度平滑性惩罚
    steer: -0.3       # 方向盘转角二次惩罚
    collision: -10.0  # 碰撞（回合结束一次性）
    finish: 20.0      # 正常跑完场景
    fail: -10.0       # 仿真失败
```

### 训练（train 段）

```yaml
train:
  total_timesteps: 200000    # 总交互步数
  buffer_size: 1000000       # 经验池大小
  learning_starts: 1000      # 攒满多少步经验后才开始训练
  batch_size: 256
  learning_rate: 0.0003
  ent_coef: "auto"
  checkpoint_freq: 10000     # 每多少步保存 checkpoint
```

## 运行

```bash
# 在项目根目录执行
python train.py

# 指定配置 / 断点续训
python train.py --config configs/sac_lasvsim.yaml
python train.py --checkpoint models/sac_lasvsim_10000_steps.zip
```

训练过程：

- 经验池未满 `learning_starts` 时只收集数据（随机动作探索），不训练
- 之后每 `train_freq` 步从经验池采样 `batch_size` 条做 `gradient_steps` 次更新
- 每 `checkpoint_freq` 步在 `save_path` 存 checkpoint，训练结束保存 `sac_lasvsim_final`
- 用 tensorboard 看曲线：`tensorboard --logdir logs`

## 扩展

- **换任务/剧本**：改 yaml 的 `scenario.task_id / record_id`；运行中换场景会自动重建仿真实例
- **改观测/奖励**：改 [obs_builder.py](gymnasium_env/envs/obs_builder.py) / [reward.py](gymnasium_env/envs/reward.py)，并在 yaml 里调整对应参数
- **并行加速**：当前为单环境串行交互；可创建多个 `LasVSimAdapter` + `SubprocVecEnv` 并行采集（每个环境对应一个仿真实例）

## 常见问题

- **提示"请先填写任务ID与剧本ID"**：确认 yaml 中 `scenario.task_id / record_id` 非 0；若已填仍报错，检查是否在项目根目录运行（`train.py` 默认按相对路径找 `configs/sac_lasvsim.yaml`），或改用 `--config` 传绝对路径
- **连接失败 / 401**：检查 token 是否过期（个人中心重新复制），endpoint 是否正确
- **车辆不动或行为异常**：`action.steer_rad_range / accel_range` 与平台车型量程不匹配，按实际控制范围调整。注意转向角单位是**弧度**，填角度值会导致转向饱和
- **每轮结束平台行为**：SDK 的 `reset` 不接收 seed，随机性来自交通流重置；若平台不支持结束后 reset，框架会自动重建仿真实例兜底