"""gymnasium 环境注册。"""
from gymnasium.envs.registration import register

register(
    id="LasVSim-v0",
    entry_point="gymnasium_env.envs.lasvsim_env:LasVSimEnv",
)
