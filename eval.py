"""训练好的 SAC 模型在 LasVSim 平台上的推理评估脚本。

流程：加载模型 -> 每回合 env.reset() -> 循环 model.predict(deterministic=True) + env.step
     -> 记录指标 -> 汇总打印 + 写入 CSV。

用法：
    python eval.py --model models/sac_lasvsim_final.zip --episodes 10
    python eval.py --model models/sac_lasvsim_10000_steps.zip --config configs/sac_lasvsim.yaml \
                   --episodes 20 --out eval_results.csv

注意：推理必须使用与训练时一致的配置文件（观测维度由 yaml 决定）。
"""
import argparse
import csv
import os

import numpy as np
from stable_baselines3 import SAC

from gymnasium_env.config import load_cfg
from gymnasium_env.envs.lasvsim_adapter import LasVSimAdapter
from gymnasium_env.envs.lasvsim_env import LasVSimEnv
from gymnasium_env.envs.reward import RewardFn

CSV_COLUMNS = [
    "episode", "steps", "sim_time", "total_reward", "mean_speed",
    "mean_abs_t_error", "max_abs_t_error", "collision", "reason", "replay_url",
]

REASON_LABELS = {
    "finished": "正常跑完",
    "failed": "仿真失败",
    "collision": "碰撞",
    "timeout": "超时截断",
    "error": "SDK异常",
    "unknown": "未知",
}


def make_env(cfg):
    adapter = LasVSimAdapter(cfg)
    reward_fn = RewardFn(cfg.reward)
    return LasVSimEnv(cfg, adapter, reward_fn)


def _term_reason(info: dict, terminated: bool, truncated: bool) -> str:
    """根据最后一步 info 判定终止原因。"""
    if info.get("error"):
        return "error"
    flags = info.get("done_flags", {}) or {}
    if flags.get("collision"):
        return "collision"
    if flags.get("finished"):
        return "finished"
    if flags.get("failed"):
        return "failed"
    if truncated and not terminated:
        return "timeout"
    return "unknown"


def run_episode(model: SAC, env: LasVSimEnv, ep_idx: int) -> dict:
    """运行一个回合，返回指标 dict。"""
    obs, _ = env.reset()
    speeds, t_errs = [], []
    total_reward = 0.0
    steps = 0
    terminated, truncated = False, False
    info = {}

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        ego = info.get("ego_state", {}) if isinstance(info, dict) else {}
        speeds.append(float(ego.get("u", 0.0)))
        t_errs.append(abs(float(ego.get("t_error", 0.0))))

    flags = info.get("done_flags", {}) if isinstance(info, dict) else {}
    replay_url = ""
    try:
        replay_url = env.adapter.get_replay_url() or ""
    except Exception:
        pass

    return {
        "episode": ep_idx,
        "steps": steps,
        "sim_time": round(float(info.get("sim_time", 0.0)), 2),
        "total_reward": round(total_reward, 3),
        "mean_speed": round(float(np.mean(speeds)) if speeds else 0.0, 3),
        "mean_abs_t_error": round(float(np.mean(t_errs)) if t_errs else 0.0, 4),
        "max_abs_t_error": round(float(np.max(t_errs)) if t_errs else 0.0, 4),
        "collision": bool(flags.get("collision", False)),
        "reason": _term_reason(info, terminated, truncated),
        "replay_url": replay_url,
    }


def print_row(row: dict):
    print(
        f"回合 {row['episode']:>3} | 步数 {row['steps']:>4} | 时长 {row['sim_time']:>7.2f}s | "
        f"总奖励 {row['total_reward']:>9.3f} | 均速 {row['mean_speed']:>5.2f} | "
        f"均|t_err| {row['mean_abs_t_error']:.3f} | 最大|t_err| {row['max_abs_t_error']:.3f} | "
        f"碰撞 {'是' if row['collision'] else '否'} | {REASON_LABELS.get(row['reason'], row['reason'])}"
    )
    if row["replay_url"]:
        print(f"      回放: {row['replay_url']}")


def print_summary(rows: list):
    n = len(rows)
    finished = sum(1 for r in rows if r["reason"] == "finished")
    rewards = [r["total_reward"] for r in rows]
    t_errs = [r["mean_abs_t_error"] for r in rows]
    durations = [r["sim_time"] for r in rows]
    print("\n================ 评估汇总 ================")
    print(f"回合数: {n}   成功率(正常跑完): {finished}/{n} ({finished / n * 100:.1f}%)")
    if rewards:
        print(f"平均总奖励: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"平均回合时长: {np.mean(durations):.2f}s")
        print(f"平均横向误差 |t_error|: {np.mean(t_errs):.4f} m")
    print("==========================================")


def write_csv(path: str, rows: list):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in CSV_COLUMNS})
    print(f"\n结果已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="SAC 模型推理评估")
    parser.add_argument("--model", default="models/sac_lasvsim_final.zip", help="模型 zip 路径")
    parser.add_argument("--config", default="configs/sac_lasvsim.yaml", help="配置文件路径（须与训练时一致）")
    parser.add_argument("--episodes", type=int, default=10, help="评估回合数")
    parser.add_argument("--out", default="eval_results.csv", help="CSV 输出路径")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    env = make_env(cfg)
    model = SAC.load(args.model)

    rows = []
    try:
        for i in range(args.episodes):
            row = run_episode(model, env, i)
            rows.append(row)
            print_row(row)
    finally:
        env.close()  # 释放平台侧仿真资源

    print_summary(rows)
    write_csv(args.out, rows)


if __name__ == "__main__":
    main()
