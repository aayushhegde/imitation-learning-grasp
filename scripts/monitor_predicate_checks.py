"""Check that each predicate of the strict-completion monitor rejects the failure it targets.

Each scenario runs the 13-phase reference expert with one deliberate defect, so that a
single certificate predicate should fail. The unmodified expert is the control. For every
rollout the script records the monitor's latched flags, the longest stability run, and the
first barrier contact with penetration deeper than 0.1 mm (step, contacting body and link,
and whether the cube had already crossed).

Usage (from the repository root):
    python scripts/monitor_predicate_checks.py --seeds 7 8 9 --out output/monitor_checks
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ilgrasp.experts.verified import VerifiedExpertPhase as P
from ilgrasp.experts.verified import VerifiedPickPlaceExpert
from ilgrasp.sim.verified_task import VerifiedPlacementEnv

GAIN = 20.0


def move(ee: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
    return np.r_[np.clip(GAIN * (target - ee), -1.0, 1.0), gripper].astype(np.float32)


class Scenario:
    """Wrap the expert; ``override`` may replace the expert command at any step."""

    name = "control"
    expected = "accepted"

    def __init__(self) -> None:
        self.expert = VerifiedPickPlaceExpert()
        self.mode = "expert"
        self.hold: np.ndarray | None = None

    def act(self, raw: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        phase_before = self.expert.phase
        action = self.expert.act(raw)
        main = np.asarray(raw["observation"])
        return self.override(action, phase_before, main[0:3], main[7:10],
                             np.asarray(raw["desired_goal"]), info)

    def override(self, action, phase, ee, obj, goal, info):  # noqa: ANN001
        return action


class NeverClose(Scenario):
    name = "gripper_never_closes"
    expected = "rejected: lift"

    def override(self, action, phase, ee, obj, goal, info):
        action[3] = 1.0
        return action


class StopBeforeCrossing(Scenario):
    name = "hold_after_lift"
    expected = "rejected: crossing"

    def override(self, action, phase, ee, obj, goal, info):
        if self.mode == "expert" and phase is P.PRE_CROSS:
            self.mode, self.hold = "hold", ee.copy()
        if self.mode == "hold":
            return move(ee, self.hold, -1.0)
        return action


class KeepClosedAtPlacement(Scenario):
    name = "gripper_kept_closed_at_goal"
    expected = "rejected: release"

    def override(self, action, phase, ee, obj, goal, info):
        if phase in (P.OPEN, P.RETREAT_UP, P.RETREAT_HOME, P.VERIFY):
            action[3] = -1.0
        return action


class StayAfterRelease(Scenario):
    name = "open_but_no_retreat"
    expected = "rejected: retreat"

    def override(self, action, phase, ee, obj, goal, info):
        if self.mode == "expert" and phase is P.RETREAT_UP:
            self.mode, self.hold = "hold", ee.copy()
        if self.mode == "hold":
            return move(ee, self.hold, 1.0)
        return action


class PushOutAfterRelease(Scenario):
    """After release, rise, then push the cube off the target with the open hand.

    The fingers open along the world y axis, so the hand approaches from the barrier side of
    the cube and sweeps in +y; the leading finger slides the cube about 0.15 m from the goal centre.
    """

    name = "cube_displaced_after_release"
    expected = "rejected: stability (cube leaves target)"

    def override(self, action, phase, ee, obj, goal, info):
        if self.mode == "expert" and phase is P.RETREAT_UP:
            self.mode, self.cube = "rise", obj.copy()
        if self.mode == "rise":
            target = np.array([ee[0], ee[1], 0.10])
            if abs(ee[2] - 0.10) < 0.012:
                self.mode = "behind"
            return move(ee, target, 1.0)
        if self.mode == "behind":
            target = np.array([self.cube[0], self.cube[1] - 0.07, 0.10])
            if np.linalg.norm(ee - target) < 0.012:
                self.mode = "down"
            return move(ee, target, 1.0)
        if self.mode == "down":
            target = np.array([self.cube[0], self.cube[1] - 0.07, 0.035])
            if np.linalg.norm(ee - target) < 0.018:
                self.mode = "sweep"
            return move(ee, target, 1.0)
        if self.mode == "sweep":
            # Slow sweep (at most 0.3 x 0.05 m per command) so the cube slides rather than flies.
            target = np.array([self.cube[0], self.cube[1] + 0.07, 0.035])
            if np.linalg.norm(ee - target) < 0.018:
                self.mode = "up"
            command = move(ee, target, 1.0)
            command[:3] = np.clip(command[:3], -0.3, 0.3)
            return command
        if self.mode == "up":
            target = np.array([ee[0], ee[1], 0.17])
            if abs(ee[2] - 0.17) < 0.018:
                self.mode = "home"
            return move(ee, target, 1.0)
        if self.mode == "home":
            return move(ee, np.array([0.04, 0.0, 0.20]), 1.0)
        return action


class RegraspAfterRelease(Scenario):
    """Once the stability counter has started, return to the cube and close on it."""

    name = "regrasp_after_counter_started"
    expected = "rejected: stability counter resets"

    def override(self, action, phase, ee, obj, goal, info):
        if self.mode == "expert" and info.get("stable_steps", 0) >= 5:
            self.mode = "descend"
        if self.mode == "descend":
            target = obj + np.array([0.0, 0.0, 0.015])
            if np.linalg.norm(ee - target) < 0.018:
                self.mode, self.hold = "closed", target.copy()
            return move(ee, target, 1.0)
        if self.mode == "closed":
            return move(ee, self.hold, -1.0)
        return action


class LowCrossing(Scenario):
    """Carry the lifted cube toward the goal at an end-effector height of 0.09 m.

    The barrier top is at 0.08 m and the held cube's centre sits about 0.015 m below the
    end effector, so the cube's lower face meets the barrier before it crosses.
    """

    name = "low_crossing_contact_before_crossing"
    expected = "rejected: collision"

    def override(self, action, phase, ee, obj, goal, info):
        if self.mode == "expert" and phase is P.PRE_CROSS:
            self.mode = "pre"
        if self.mode == "pre":
            target = np.array([goal[0], -0.075, 0.09])
            if np.linalg.norm(ee - target) < 0.018:
                self.mode = "cross"
            return move(ee, target, -1.0)
        if self.mode == "cross":
            return move(ee, np.array([goal[0], 0.075, 0.09]), -1.0)
        return action


class StrikeBarrierAfterRelease(Scenario):
    """After a correct release, rise vertically, touch the barrier with the hand, go home."""

    name = "robot_contact_after_crossing"
    expected = "rejected: collision only"

    def override(self, action, phase, ee, obj, goal, info):
        if self.mode == "expert" and phase is P.RETREAT_UP:
            self.mode = "rise"
        if self.mode == "rise":
            target = np.array([ee[0], ee[1], 0.17])
            if abs(ee[2] - 0.17) < 0.018:
                self.mode = "over"
            return move(ee, target, 1.0)
        if self.mode == "over":
            target = np.array([goal[0], 0.10, 0.17])
            if np.linalg.norm(ee - target) < 0.018:
                self.mode, self.k = "strike", 0
            return move(ee, target, 1.0)
        if self.mode == "strike":
            self.k += 1
            if self.k >= 15:
                self.mode = "home"
            return move(ee, np.array([goal[0], 0.0, 0.06]), 1.0)
        if self.mode == "home":
            return move(ee, np.array([0.04, 0.0, 0.20]), 1.0)
        return action


SCENARIOS: list[Callable[[], Scenario]] = [
    Scenario, NeverClose, StopBeforeCrossing, KeepClosedAtPlacement, StayAfterRelease,
    PushOutAfterRelease, RegraspAfterRelease, LowCrossing, StrikeBarrierAfterRelease,
]


def body_link_name(client: Any, body: int, link: int, names: dict[int, str]) -> str:
    body_name = names.get(body) or client.getBodyInfo(body)[1].decode()
    if link < 0:
        return body_name
    return f"{body_name}:{client.getJointInfo(body, link)[12].decode()}"


def run(scenario: Scenario, seed: int) -> dict[str, Any]:
    env = VerifiedPlacementEnv()
    try:
        env.reset(seed=seed)
        sim = env.unwrapped.sim
        client = sim.physics_client
        barrier = sim._bodies_idx["verified_barrier"]
        static_ids = {sim._bodies_idx[k] for k in ("table", "plane") if k in sim._bodies_idx}
        names = {v: k for k, v in sim._bodies_idx.items()}
        info = dict(env.last_info)
        first_contact: dict[str, Any] | None = None
        static_contacts = 0
        max_stable = 0
        resets_after_start = 0
        prev_stable = 0
        steps = 0
        for _ in range(env.config.max_episode_steps):
            action = scenario.act(env.raw_obs, info)
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            for c in client.getContactPoints(bodyB=barrier):
                other = c[1] if c[2] == barrier else c[2]
                other_link = c[3] if c[2] == barrier else c[4]
                if other in static_ids:
                    static_contacts += 1
                if first_contact is None and float(c[8]) < -1e-4:
                    first_contact = {
                        "step": steps,
                        "body_link": body_link_name(client, other, other_link, names),
                        "penetration_mm": round(-1000 * float(c[8]), 3),
                        "after_crossing": bool(info["ever_crossed"]),
                    }
            max_stable = max(max_stable, info["stable_steps"])
            if prev_stable > 0 and info["stable_steps"] == 0:
                resets_after_start += 1
            prev_stable = info["stable_steps"]
            if terminated or truncated:
                break
        return {
            "scenario": scenario.name,
            "expected": scenario.expected,
            "seed": seed,
            "steps": steps,
            "strict_success": bool(info["strict_success"]),
            "native_goal_seen": bool(info["native_success_seen"]),
            "lift": bool(info["ever_lifted"]),
            "cross": bool(info["ever_crossed"]),
            "release": bool(info["ever_released"]),
            "retreat_final": bool(info["retreated"]),
            "max_stable_steps": int(max_stable),
            "counter_resets": int(resets_after_start),
            "collision": bool(info["barrier_violation"]),
            "first_contact": first_contact,
            "static_body_contacts": static_contacts,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--out", type=Path, default=Path("output/monitor_checks"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [run(make(), seed) for make in SCENARIOS for seed in args.seeds]
    with open(args.out / "monitor_checks.json", "w") as f:
        json.dump({"platform": platform.platform(), "python": platform.python_version(),
                   "rows": rows}, f, indent=2)
    with open(args.out / "monitor_checks.csv", "w", newline="") as f:
        fields = [k for k in rows[0] if k != "first_contact"] + [
            "contact_step", "contact_body_link", "contact_after_crossing"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            fc = r["first_contact"] or {}
            out = {k: v for k, v in r.items() if k != "first_contact"}
            out.update(contact_step=fc.get("step"), contact_body_link=fc.get("body_link"),
                       contact_after_crossing=fc.get("after_crossing"))
            writer.writerow(out)
    for r in rows:
        fc = r["first_contact"]
        print(f"{r['scenario']:38s} seed={r['seed']:<5d} strict={int(r['strict_success'])} "
              f"lift={int(r['lift'])} cross={int(r['cross'])} rel={int(r['release'])} "
              f"ret={int(r['retreat_final'])} maxstab={r['max_stable_steps']:>3d} "
              f"resets={r['counter_resets']} coll={int(r['collision'])} "
              f"contact={fc['body_link'] + '@' + str(fc['step']) + (' post' if fc['after_crossing'] else ' pre') if fc else '-'} "
              f"static={r['static_body_contacts']}")


if __name__ == "__main__":
    main()
