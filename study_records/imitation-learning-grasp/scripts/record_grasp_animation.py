#!/usr/bin/env python3
"""Record a real PyBullet Panda grasp rollout as an annotated GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ilgrasp.experts import ScriptedExpert
from ilgrasp.sim import GraspEnv

PHASE_COLORS = {
    "approach": "#4C78A8",
    "descend": "#72B7B2",
    "grasp": "#F58518",
    "lift": "#E45756",
    "transport": "#54A24B",
    "place": "#B279A2",
    "hold": "#FF9DA6",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _annotate(
    frame: np.ndarray,
    *,
    phase: str,
    step: int,
    object_goal_distance: float,
    success: bool = False,
    width: int = 600,
) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((12, 12, width - 12, 70), radius=10, fill=(15, 24, 39, 205))
    draw.rounded_rectangle((25, 24, 145, 57), radius=8, fill=PHASE_COLORS.get(phase, "#777777"))
    draw.text((85, 40), phase.upper(), anchor="mm", fill="white", font=_font(15))
    draw.text(
        (160, 27),
        f"Step {step:02d}   object → goal: {object_goal_distance:.3f} m",
        fill="white",
        font=_font(15),
    )
    if success:
        draw.rounded_rectangle(
            (width // 2 - 105, height - 62, width // 2 + 105, height - 18),
            radius=10,
            fill=(25, 135, 84, 225),
        )
        draw.text(
            (width // 2, height - 40),
            "TASK SUCCESS",
            anchor="mm",
            fill="white",
            font=_font(20),
        )
    return image


def record_expert_gif(
    output: Path,
    *,
    seed: int = 3,
    duration_ms: int = 170,
    width: int = 600,
) -> dict[str, int | bool | str]:
    """Render an expert rollout and return a small recording summary."""

    output.parent.mkdir(parents=True, exist_ok=True)
    env = GraspEnv(render_mode="rgb_array")
    expert = ScriptedExpert()
    frames: list[Image.Image] = []
    success = False
    step = 0

    try:
        env.reset(seed=seed)
        expert.reset()
        while True:
            if env.raw_obs is None:
                raise RuntimeError("Simulator did not provide a raw observation")
            phase = expert.phase.value
            action = expert.act(env.raw_obs)
            _, _, terminated, truncated, info = env.step(action)
            step += 1

            main = np.asarray(env.raw_obs["observation"])
            object_position = main[7:10]
            goal_position = np.asarray(env.raw_obs["desired_goal"])
            distance = float(np.linalg.norm(object_position - goal_position))
            success = bool(info.get("is_success", False))
            frames.append(
                _annotate(
                    env.render(),
                    phase=phase,
                    step=step,
                    object_goal_distance=distance,
                    success=success,
                    width=width,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    if success:
        frames.extend([frames[-1].copy() for _ in range(7)])
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {"seed": seed, "steps": step, "success": success, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/assets/expert_grasp_3d.gif"),
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--duration-ms", type=int, default=170)
    parser.add_argument("--width", type=int, default=600)
    args = parser.parse_args()
    print(
        record_expert_gif(
            args.output,
            seed=args.seed,
            duration_ms=args.duration_ms,
            width=args.width,
        )
    )


if __name__ == "__main__":
    main()
