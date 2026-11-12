import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gymnasium_env.envs.action_mapper import ActionMapper
from gymnasium_env.envs.obs_builder import ObsBuilder
from gymnasium_env.envs.termination import compute_termination


class LasVSimEnv(gym.Env):

    def __init__(self, cfg, adapter, reward_fn):
        self.cfg = cfg
        self.adapter = adapter
        self.reward_fn = reward_fn
        self.action_mapper = ActionMapper(cfg.action)
        self.obs_builder = ObsBuilder(cfg.obs)
        self.action_space = spaces.Box(
            low=np.array([cfg.action.steer_cmd_range[0], cfg.action.accel_cmd_range[0]], np.float32),
            high=np.array([cfg.action.steer_cmd_range[1], cfg.action.accel_cmd_range[1]], np.float32),
            dtype=np.float32,
        )
        shapes = self.obs_builder.shapes()
        self.observation_space = spaces.Dict({
            k: spaces.Box(-np.inf, np.inf, s, np.float32) for k, s in shapes.items()
        })

        self._prev_obs = None
        self._sim_time = 0.0
        self._done = True
        self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            self.adapter.connect()
            self._connected = True

    def _make_info(self, res) -> dict:
        return {
            "sim_time": res.sim_time,
            "done_flags": dict(res.done_flags),
            **dict(res.info),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_connected()
        options = options or {}
        task_id = options.get("task_id", self.cfg.scenario.task_id)
        record_id = options.get("record_id", self.cfg.scenario.record_id)

        res = self.adapter.reset(task_id, record_id, seed=seed)
        self._sim_time = res.sim_time
        obs = self.obs_builder.build(res.raw_obs)
        self._prev_obs = obs
        self._done = False
        return obs, self._make_info(res)

    def step(self, action):
        if self._done:
            raise RuntimeError("episode 已结束，请先调用 reset()")

        action = np.asarray(action, dtype=np.float32)
        steer_deg, lon_acc = self.action_mapper.to_physical(action)
        self.adapter.set_control(steer_deg, lon_acc)

        try:
            res = self.adapter.step()
        except Exception as e:
            # 通信/SDK 异常 -> truncated，避免训练循环崩溃
            self.adapter.end_episode()
            self._done = True
            info = {"sim_time": self._sim_time, "done_flags": {}, "error": repr(e)}
            obs = self._prev_obs
            return obs, 0.0, False, True, info

        self._sim_time = res.sim_time
        obs = self.obs_builder.build(res.raw_obs)

        reward = float(self.reward_fn.compute(obs, action, self._prev_obs, res.done_flags, res.info))
        term = compute_termination(res.is_running, self._sim_time, self.cfg.term, res.done_flags)

        if term.terminated or term.truncated:
            self.adapter.end_episode()
            self._done = True

        info = self._make_info(res)
        self._prev_obs = obs
        return obs, reward, term.terminated, term.truncated, info

    def close(self):
        try:
            self.adapter.close()
        finally:
            self._connected = False
