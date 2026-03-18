"""
Root conftest.py — project-wide pytest configuration.

• Excludes manual/integration scripts that contain bare exit() calls from
  automatic test collection (they are not meant to be run by pytest).
• Adds the project root to sys.path so that `services.*` and `agents.*`
  imports resolve correctly regardless of how pytest is invoked.
"""

import sys
from pathlib import Path

# Ensure project root is always on the path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Tell pytest to ignore manual scripts that use exit() at module level.
# Add relative paths (from rootdir) as strings.
# ---------------------------------------------------------------------------
collect_ignore: list[str] = [
    "agents/course_ingestion/tests/test_ingestion.py",  # manual PDF ingestion script
]

collect_ignore_glob: list[str] = [
    "**/demo_*.py",  # interactive demo scripts
    "**/test_*_manual.py",  # explicitly manual scripts
]
