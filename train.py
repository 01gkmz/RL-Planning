"""基于 stable-baselines3 SAC 的 LasVSim 强化学习训练入口。

训练流程：
    1. 环境通过 lasvsim Open API 与千行仿真平台交互（见 gymnasium_env/envs/lasvsim_adapter.py）
    2. 每步交互产生的 (obs, action, reward, next_obs, done) 存入 SAC 自带的经验池 (ReplayBuffer)
    3. 经验积累到 learning_starts 步后，SAC 开始从经验池采样并更新网络
    4. 之后边交互边训练（train_freq 步训练 gradient_steps 次）

用法：
    python train.py --config configs/sac_lasvsim.yaml
    python train.py --config configs/sac_lasvsim.yaml --checkpoint models/sac_lasvsim_10000_steps.zip
"""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from gymnasium_env.config import load_cfg
from gymnasium_env.envs.lasvsim_adapter import LasVSimAdapter
from gymnasium_env.envs.lasvsim_env import LasVSimEnv
from gymnasium_env.envs.reward import RewardFn


def make_env(cfg):
    adapter = LasVSimAdapter(cfg)
    reward_fn = RewardFn(cfg.reward)
    return LasVSimEnv(cfg, adapter, reward_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/lisixu/rl-planning/configs/sac_lasvsim.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", default=None, help="从已有模型继续训练")
    
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    print("DEBUG scenario:", cfg.scenario)

    if cfg.scenario.task_id == 0 or cfg.scenario.record_id == 0:
        print("[警告] scenario.task_id / record_id 为 0，请先在配置文件中填写千行平台的真实任务ID与剧本ID")

    env = make_env(cfg)
    train_cfg = cfg.train
    os.makedirs(train_cfg.save_path, exist_ok=True)

    if args.checkpoint:
        model = SAC.load(args.checkpoint, env=env, tensorboard_log=train_cfg.log_dir)
        print(f"从 checkpoint 继续训练: {args.checkpoint}")
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=train_cfg.learning_rate,
            buffer_size=train_cfg.buffer_size,          # 经验池大小
            learning_starts=train_cfg.learning_starts,  # 积累多少步后开始训练
            batch_size=train_cfg.batch_size,
            tau=train_cfg.tau,
            gamma=train_cfg.gamma,
            train_freq=train_cfg.train_freq,
            gradient_steps=train_cfg.gradient_steps,
            ent_coef=train_cfg.ent_coef,
            policy_kwargs=dict(net_arch=dict(pi=train_cfg.net_arch, qf=train_cfg.net_arch)),
            tensorboard_log=train_cfg.log_dir,
            verbose=1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg.checkpoint_freq,
        save_path=train_cfg.save_path,
        name_prefix="sac_lasvsim",
    )

    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=checkpoint_callback,
        log_interval=10,
    )

    final_path = os.path.join(train_cfg.save_path, "sac_lasvsim_final")
    model.save(final_path)
    print(f"训练完成，模型已保存: {final_path}")
    env.close()


if __name__ == "__main__":
    main()
