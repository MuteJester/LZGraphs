"""Terminal rendering layer for the LZGraphs CLI.

``_caps`` is the single place in this package that reads ``os.environ`` or
calls ``isatty``/``shutil.get_terminal_size``; later tasks (escape sequences,
widgets, the plain and rich renderers) ask a ``Capabilities`` instance rather
than reading the environment themselves. This module currently only exports
what Task 1 produces; later tasks add the rest of the public surface
(``ui()``, the ``Ui`` protocol) here.
"""
from __future__ import annotations

from ._caps import Capabilities, detect, resolve_mode

__all__ = [
    "Capabilities",
    "detect",
    "resolve_mode",
]
