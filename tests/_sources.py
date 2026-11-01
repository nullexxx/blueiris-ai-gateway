"""Shared source readers for the static contract tests.

The PTZ tracker used to live in a single ``tracker.py``. It is now split across
several modules (``tracker.py`` plus the ``tracker_*`` mixins/helpers), so the
contract tests read the combined source of all of them instead of one file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

TRACKER_MODULES = (
    "tracker.py",
    "tracker_config.py",
    "tracker_models.py",
    "tracker_policy.py",
    "tracker_servo.py",
    "tracker_calibration.py",
)

# Inserted between modules so source slices never silently run from one file into the next.
MODULE_END = "# ==== end of tracker module ===="

_FUTURE_IMPORT = re.compile(r"^from __future__ import annotations[ \t]*$", re.M)
# Start of the next definition at class-member or module level.
_NEXT_DEF = re.compile(r"^(?:    )?(?:async def |def |class |@)", re.M)


def read(name: str) -> str:
    return (ROOT / name).read_text()


@lru_cache(maxsize=None)
def tracker_source() -> str:
    """Return the combined, parseable source of every tracker module."""
    parts = []
    for name in TRACKER_MODULES:
        text = _FUTURE_IMPORT.sub("", read(name))
        parts.append(f"{text}\n{MODULE_END} ({name})\n")
    return "from __future__ import annotations\n" + "\n".join(parts)


TRACKER = tracker_source()


def tracker_block(start: str, end: str, source: str = TRACKER) -> str:
    """Slice ``source`` from ``start`` up to ``end``.

    The slice never crosses a module boundary. When ``end`` no longer follows
    ``start`` in the same module (because the methods were moved into different
    files), the slice ends at the end of the definition that ``start`` opens.
    """
    a = source.index(start)
    search_from = a + len(start)
    boundary = source.find(MODULE_END, search_from)
    if boundary == -1:
        boundary = len(source)
    b = source.find(end, search_from, boundary)
    if b != -1:
        return source[a:b]
    nxt = _NEXT_DEF.search(source, search_from, boundary)
    return source[a: nxt.start() if nxt else boundary]
