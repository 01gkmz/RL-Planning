"""终止判断：仿真结束 / 碰撞 / 超时 -> terminated / truncated。"""
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class TermResult:
    terminated: bool = False
    truncated: bool = False
    reason: str = ""


def compute_termination(
    is_running: bool,
    sim_time: float,
    cfg,
    done_flags: Optional[Dict[str, bool]] = None,
) -> TermResult:
    done_flags = done_flags or {}

    if not is_running:
        if done_flags.get("failed"):
            return TermResult(True, False, "sim_failed")
        return TermResult(True, False, "sim_finished")

    if done_flags.get("collision") and cfg.collision_terminate:
        return TermResult(True, False, "collision")

    if sim_time >= cfg.max_episode_time:
        return TermResult(False, True, "timeout")

    return TermResult(False, False, "")
