"""Compose a two-camera, telemetry-rich GIF of verified placement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ilgrasp.experts import VerifiedExpertPhase, VerifiedPickPlaceExpert
from ilgrasp.sim import VerifiedPlacementEnv

PHASES = [phase.value for phase in VerifiedExpertPhase]
PHASE_SHORT = [
    "APP",
    "DOWN",
    "CLOSE",
    "TEST",
    "LIFT",
    "PRE",
    "CROSS",
    "GOAL",
    "LOWER",
    "OPEN",
    "UP",
    "HOME",
    "VERIFY",
]
PHASE_COLORS = {
    "approach": "#4C78A8",
    "descend": "#72B7B2",
    "close": "#F58518",
    "test_grasp": "#ECA82C",
    "lift_clear": "#E45756",
    "pre_cross": "#A86BC1",
    "cross_barrier": "#7B61A8",
    "above_goal": "#54A24B",
    "lower": "#8BC34A",
    "open": "#FF9DA6",
    "retreat_up": "#B279A2",
    "retreat_home": "#6F4E7C",
    "verify": "#2A9D8F",
}


@dataclass(frozen=True)
class RecordingResult:
    output: Path
    seed: int
    steps: int
    strict_success: bool
    native_success_seen: bool
    min_crossing_clearance: float | None
    final_goal_error: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "seed": self.seed,
            "steps": self.steps,
            "strict_success": self.strict_success,
            "native_success_seen": self.native_success_seen,
            "min_crossing_clearance": self.min_crossing_clearance,
            "final_goal_error": self.final_goal_error,
        }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    names.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(image).convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=14, fill="#172235", outline="#33435C", width=2)


def _status_row(
    draw: ImageDraw.ImageDraw,
    y: int,
    label: str,
    active: bool,
    detail: str = "",
) -> None:
    color = "#35C48D" if active else "#53627A"
    draw.rounded_rectangle((882, y, 900, y + 18), radius=4, fill=color)
    draw.text((912, y - 2), label, fill="#F7FAFC", font=_font(17, bold=True))
    if detail:
        right = draw.textbbox((0, 0), detail, font=_font(15))[2]
        draw.text((1235 - right, y), detail, fill="#AFC0D8", font=_font(15))


def _compose_frame(
    main_frame: np.ndarray,
    inset_frame: np.ndarray,
    *,
    phase: str,
    step: int,
    info: dict[str, Any],
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), "#0B1220")
    draw = ImageDraw.Draw(canvas)

    draw.text((24, 18), "VERIFIED PICK · CROSS · PLACE", fill="#F8FAFC", font=_font(29, True))
    phase_color = PHASE_COLORS[phase]
    phase_label = phase.replace("_", " ").upper()
    label_width = max(180, draw.textbbox((0, 0), phase_label, font=_font(18, True))[2] + 44)
    draw.rounded_rectangle((810 - label_width, 17, 810, 58), radius=11, fill=phase_color)
    draw.text(
        (810 - label_width / 2, 37),
        phase_label,
        anchor="mm",
        fill="white",
        font=_font(18, True),
    )
    draw.text((835, 24), f"STEP {step:03d}", fill="#AFC0D8", font=_font(20, True))
    native_color = "#F2B84B" if info["native_success_seen"] else "#53627A"
    strict_color = "#35C48D" if info["strict_success"] else "#53627A"
    draw.rounded_rectangle((980, 19, 1104, 56), radius=10, fill=native_color)
    draw.text((1042, 38), "NATIVE HIT", anchor="mm", fill="white", font=_font(14, True))
    draw.rounded_rectangle((1118, 19, 1256, 56), radius=10, fill=strict_color)
    draw.text((1187, 38), "VERIFIED", anchor="mm", fill="white", font=_font(14, True))

    _panel(draw, (18, 76, 842, 632))
    canvas.paste(_fit(main_frame, (810, 540)), (25, 83))
    draw.rounded_rectangle((25, 83, 835, 623), radius=10, outline="#6D7F99", width=2)
    draw.text((38, 94), "MAIN CAMERA", fill="white", font=_font(15, True), stroke_width=2, stroke_fill="#111827")

    _panel(draw, (858, 76, 1262, 632))
    canvas.paste(_fit(inset_frame, (372, 222)), (874, 92))
    draw.rounded_rectangle((874, 92, 1246, 314), radius=9, outline="#6D7F99", width=2)
    inset_name = "GRASP / RELEASE DETAIL" if phase in {
        "descend", "close", "test_grasp", "lower", "open"
    } else "TOP VIEW · BARRIER CROSSING"
    draw.text((887, 102), inset_name, fill="white", font=_font(14, True), stroke_width=2, stroke_fill="#111827")

    draw.text((880, 330), "COMPLETION CERTIFICATE", fill="#AFC0D8", font=_font(16, True))
    _status_row(draw, 359, "Cube lifted", bool(info["ever_lifted"]))
    _status_row(draw, 389, "Barrier crossed", bool(info["ever_crossed"]))
    _status_row(draw, 419, "Gripper released", bool(info["ever_released"]))
    _status_row(draw, 449, "Arm retreated", bool(info["retreated"]))
    _status_row(
        draw,
        479,
        "Cube stable",
        int(info["stable_steps"]) >= int(info["required_stable_steps"]),
        f"{info['stable_steps']}/{info['required_stable_steps']}",
    )
    _status_row(draw, 509, "Collision-free", bool(info["collision_free"]))

    draw.text((882, 546), "goal error", fill="#8FA3BF", font=_font(14))
    draw.text((1010, 542), f"{100 * info['goal_error']:.1f} cm", fill="#F8FAFC", font=_font(18, True))
    draw.text((882, 576), "cube speed", fill="#8FA3BF", font=_font(14))
    draw.text((1010, 572), f"{100 * info['object_speed']:.2f} cm/s", fill="#F8FAFC", font=_font(18, True))
    draw.text((1135, 546), "gripper", fill="#8FA3BF", font=_font(14))
    draw.text((1135, 570), f"{100 * info['gripper_width']:.1f} cm", fill="#F8FAFC", font=_font(18, True))

    # Bottom phase rail: equal-width roadmap with the current phase highlighted.
    rail_y0, rail_y1 = 655, 702
    segment = 1238 / len(PHASES)
    current = PHASES.index(phase)
    for i, (name, short) in enumerate(zip(PHASES, PHASE_SHORT)):
        x0 = int(21 + i * segment)
        x1 = int(21 + (i + 1) * segment - 3)
        fill = PHASE_COLORS[name] if i <= current else "#253249"
        outline = "#FFFFFF" if i == current else "#3B4B64"
        draw.rounded_rectangle((x0, rail_y0, x1, rail_y1), radius=7, fill=fill, outline=outline, width=2)
        draw.text(
            ((x0 + x1) / 2, (rail_y0 + rail_y1) / 2),
            short,
            anchor="mm",
            fill="white" if i <= current else "#8FA3BF",
            font=_font(12, True),
        )

    if info["strict_success"]:
        draw.rounded_rectangle((260, 270, 600, 430), radius=24, fill="#168A61", outline="#7FFFD4", width=4)
        draw.text((430, 321), "TASK COMPLETE", anchor="mm", fill="white", font=_font(34, True))
        draw.text((430, 365), "released · retreated · stable", anchor="mm", fill="#D7FFF1", font=_font(19, True))
        draw.text((430, 397), "strict certificate passed", anchor="mm", fill="#D7FFF1", font=_font(16))

    return canvas


def record_verified_expert_gif(
    output: str | Path,
    *,
    seed: int = 7,
    frame_duration_ms: int = 110,
) -> RecordingResult:
    """Record a full expert rollout; success appears only after stability proof."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = VerifiedPlacementEnv(render_mode="rgb_array")
    expert = VerifiedPickPlaceExpert()
    captures: list[tuple[np.ndarray, np.ndarray, str, int, dict[str, Any]]] = []
    info: dict[str, Any] = {}

    try:
        env.reset(seed=seed)
        expert.reset()
        for step in range(1, env.config.max_episode_steps + 1):
            if env.raw_obs is None:
                raise RuntimeError("raw_obs unavailable during recording")
            phase = expert.phase.value
            action = expert.act(env.raw_obs)
            _, _, terminated, truncated, info = env.step(action)

            main = np.asarray(env.raw_obs["observation"])
            ee, obj = main[0:3], main[7:10]
            main_frame = env.render_view(
                width=720,
                height=480,
                target_position=np.array([-0.12, 0.0, 0.14]),
                distance=1.18,
                yaw=55,
                pitch=-25,
            )
            if phase in {"descend", "close", "test_grasp", "lower", "open"}:
                inset_target = 0.5 * (ee + obj)
                inset_frame = env.render_view(
                    width=480,
                    height=300,
                    target_position=inset_target,
                    distance=0.46,
                    yaw=42,
                    pitch=-24,
                )
            else:
                inset_frame = env.render_view(
                    width=480,
                    height=300,
                    target_position=np.array([0.0, 0.0, 0.04]),
                    distance=0.78,
                    yaw=90,
                    pitch=-84,
                )
            captures.append((main_frame, inset_frame, phase, step, dict(info)))
            if terminated or truncated:
                break
    finally:
        env.close()

    if not captures:
        raise RuntimeError("Recording produced no frames")
    frames = [
        _compose_frame(main, inset, phase=phase, step=step, info=frame_info)
        for main, inset, phase, step, frame_info in captures
    ]
    durations: list[int] = []
    previous_phase: str | None = None
    for _, _, phase, _, _ in captures:
        durations.append(280 if phase != previous_phase else frame_duration_ms)
        previous_phase = phase
    if bool(info.get("strict_success", False)):
        for _ in range(8):
            frames.append(frames[-1].copy())
            durations.append(180)

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return RecordingResult(
        output=output,
        seed=seed,
        steps=int(info.get("step", len(captures))),
        strict_success=bool(info.get("strict_success", False)),
        native_success_seen=bool(info.get("native_success_seen", False)),
        min_crossing_clearance=info.get("min_crossing_clearance"),
        final_goal_error=float(info.get("goal_error", float("nan"))),
    )
