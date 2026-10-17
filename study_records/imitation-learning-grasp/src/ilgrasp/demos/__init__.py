# Public entry point for the demos package: `from ilgrasp.demos import load_demos`
# instead of reaching into loader.py directly, so the internal module layout
# can change without breaking callers elsewhere in the pipeline.
from .loader import Demo, load_demos

__all__ = ["Demo", "load_demos"]
