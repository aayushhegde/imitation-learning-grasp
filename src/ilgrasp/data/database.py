"""SQLite storage for state-action demonstrations.

Episodes and steps are separate tables so students can ask useful SQL questions
without loading the whole dataset. Arrays are encoded as JSON because they are
small, readable, and easy to inspect with any SQLite browser.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ilgrasp.sim import ACTION_DIM, OBS_DIM, Episode


@dataclass
class TrainingData:
    observations: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray
    phases: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.observations)
        if self.observations.shape != (n, OBS_DIM):
            raise ValueError(f"observations must have shape (N, {OBS_DIM})")
        if self.actions.shape != (n, ACTION_DIM):
            raise ValueError(f"actions must have shape (N, {ACTION_DIM})")
        if self.episode_ids.shape != (n,) or self.phases.shape != (n,):
            raise ValueError("episode_ids and phases must have one value per sample")

    def __len__(self) -> int:
        return len(self.observations)


class DemoDatabase:
    """Context-managed database with atomic episode writes."""

    def __init__(self, path: str | Path = "demos.db") -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                seed        INTEGER,
                success     INTEGER NOT NULL CHECK (success IN (0, 1)),
                usable      INTEGER NOT NULL DEFAULT 1 CHECK (usable IN (0, 1)),
                n_steps     INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS steps (
                episode_id INTEGER NOT NULL,
                step       INTEGER NOT NULL,
                observation TEXT NOT NULL,
                action     TEXT NOT NULL,
                reward     REAL NOT NULL,
                phase      TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (episode_id, step),
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success);
            CREATE INDEX IF NOT EXISTS idx_steps_phase ON steps(phase);
            """
        )
        # Forward-compatible migration for databases created by pipeline v0.1.
        episode_columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(episodes)").fetchall()
        }
        if "usable" not in episode_columns:
            self.connection.execute(
                "ALTER TABLE episodes ADD COLUMN usable INTEGER NOT NULL DEFAULT 1"
            )
        self.connection.commit()

    def save_episode(self, episode: Episode) -> int:
        """Insert one complete episode in a transaction and return its id."""

        phases = episode.phases or [""] * len(episode)
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO episodes(seed, success, usable, n_steps, total_reward)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    episode.seed,
                    int(episode.success),
                    int(bool(episode.usable_for_training)),
                    len(episode),
                    episode.total_reward,
                ),
            )
            episode_id = int(cursor.lastrowid)
            rows = [
                (
                    episode_id,
                    step,
                    json.dumps(obs.tolist(), separators=(",", ":")),
                    json.dumps(action.tolist(), separators=(",", ":")),
                    float(reward),
                    phase,
                )
                for step, (obs, action, reward, phase) in enumerate(
                    zip(episode.observations, episode.actions, episode.rewards, phases)
                )
            ]
            self.connection.executemany(
                """
                INSERT INTO steps(episode_id, step, observation, action, reward, phase)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return episode_id

    def load_training_data(
        self, successful_only: bool = True, max_episodes: int | None = None
    ) -> TrainingData:
        """Return ordered arrays, optionally limited to the first N episodes."""

        where = "WHERE e.success = 1 AND e.usable = 1" if successful_only else "WHERE 1 = 1"
        params: tuple[int, ...] = ()
        episode_limit = ""
        if max_episodes is not None:
            if max_episodes <= 0:
                raise ValueError("max_episodes must be positive")
            episode_limit = "LIMIT ?"
            params = (max_episodes,)

        rows = self.connection.execute(
            f"""
            WITH selected AS (
                SELECT episode_id
                FROM episodes e
                {where}
                ORDER BY episode_id
                {episode_limit}
            )
            SELECT s.episode_id, s.observation, s.action, s.phase
            FROM steps s
            JOIN selected x ON x.episode_id = s.episode_id
            ORDER BY s.episode_id, s.step
            """,
            params,
        ).fetchall()
        if not rows:
            scope = "successful " if successful_only else ""
            raise ValueError(f"Database contains no {scope}demonstration steps")

        episode_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
        observations = np.asarray([json.loads(row[1]) for row in rows], dtype=np.float32)
        actions = np.asarray([json.loads(row[2]) for row in rows], dtype=np.float32)
        phases = np.asarray([row[3] for row in rows], dtype=str)
        return TrainingData(observations, actions, episode_ids, phases)

    def summary(self) -> dict[str, int | float]:
        episodes, successes, usable, steps = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(success), 0),
                   COALESCE(SUM(success * usable), 0), COALESCE(SUM(n_steps), 0)
            FROM episodes
            """
        ).fetchone()
        rate = float(successes / episodes) if episodes else 0.0
        return {
            "episodes": int(episodes),
            "successful_episodes": int(successes),
            "usable_demonstrations": int(usable),
            "steps": int(steps),
            "expert_success_rate": rate,
        }

    def export_npz(self, path: str | Path, max_episodes: int | None = None) -> None:
        data = self.load_training_data(successful_only=True, max_episodes=max_episodes)
        np.savez_compressed(
            path,
            observations=data.observations,
            actions=data.actions,
            episode_ids=data.episode_ids,
            phases=data.phases,
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DemoDatabase":  # noqa: PYI034 -- Python 3.10 compatible
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
