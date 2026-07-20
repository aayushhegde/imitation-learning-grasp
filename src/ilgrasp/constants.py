"""Shared constants for the ilgrasp environment.

Single source of truth for the observation/action contract. Anything read by
more than one module (data collection, evaluation, expert controller) lives
here so the numbers can't silently disagree.
"""

# --- Observation / action contract (the network's input & output sizes) ---
RAW_OBS_DIM = 19
GOAL_DIM = 3
OBS_DIM = RAW_OBS_DIM + GOAL_DIM
ACTION_DIM = 4

# --- Indexing into the observation ---
OBJ_POS_SLICE = slice(7, 10)

# --- Episode bounds ---
MAX_EPISODE_STEPS = 50