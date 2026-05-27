"""Snapshot helpers for Tier 1 library-level regression tests.

A snapshot is a JSON file under ``tests/regression/snapshots/`` whose
contents are the canonical output of a library calculation on a fixed
input. Tests assert the current output matches the snapshot.

To intentionally update a snapshot after a deliberate library change,
set the ``LZG_REGEN_SNAPSHOTS`` environment variable:

    LZG_REGEN_SNAPSHOTS=1 pytest tests/regression/

Comparison policy:

* dicts:  same keys, values compared recursively
* lists:  same length, elements compared positionally
* floats: ``math.isclose(a, b, rel_tol=rtol, abs_tol=atol)`` with sane
          defaults; ``NaN == NaN`` (so log_prob=-inf legitimately
          compares equal to itself), ``inf == inf`` exactly
* ints, strs, bools, None: exact equality
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

# Default tolerances for MC / floating-point outputs.
DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 1e-9


def _regen_mode() -> bool:
    return os.environ.get("LZG_REGEN_SNAPSHOTS", "") not in ("", "0", "false", "False")


def _to_json_value(obj: Any) -> Any:
    """Coerce numpy / float-special values to JSON-safe forms."""
    if isinstance(obj, dict):
        return {str(k): _to_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_value(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__nan__"
        if math.isinf(obj):
            return "__inf__" if obj > 0 else "__-inf__"
        return obj
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    # numpy ndarray (any size): convert to list and recurse
    if hasattr(obj, "tolist") and hasattr(obj, "shape"):
        return _to_json_value(obj.tolist())
    # numpy scalars (size-1, e.g. np.float64)
    if hasattr(obj, "item"):
        return _to_json_value(obj.item())
    raise TypeError(f"Cannot serialize {type(obj).__name__}: {obj!r}")


def _from_json_value(obj: Any) -> Any:
    """Inverse of _to_json_value: restore float specials."""
    if isinstance(obj, dict):
        return {k: _from_json_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json_value(v) for v in obj]
    if obj == "__nan__":
        return float("nan")
    if obj == "__inf__":
        return float("inf")
    if obj == "__-inf__":
        return float("-inf")
    return obj


def _compare(path: str, expected: Any, actual: Any, atol: float, rtol: float) -> None:
    if type(expected) is not type(actual):
        # Allow int/float cross-comparison
        if not (isinstance(expected, (int, float)) and isinstance(actual, (int, float))):
            raise AssertionError(
                f"{path}: type mismatch — expected {type(expected).__name__}, "
                f"got {type(actual).__name__}"
            )

    if isinstance(expected, dict):
        if set(expected.keys()) != set(actual.keys()):
            missing = set(expected.keys()) - set(actual.keys())
            extra = set(actual.keys()) - set(expected.keys())
            raise AssertionError(
                f"{path}: key mismatch — missing={sorted(missing)} extra={sorted(extra)}"
            )
        for k in expected:
            _compare(f"{path}.{k}", expected[k], actual[k], atol, rtol)
        return

    if isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(
                f"{path}: list length mismatch — expected {len(expected)}, "
                f"got {len(actual)}"
            )
        for i, (e, a) in enumerate(zip(expected, actual)):
            _compare(f"{path}[{i}]", e, a, atol, rtol)
        return

    if isinstance(expected, float) or isinstance(actual, float):
        e = float(expected)
        a = float(actual)
        # NaN matches NaN; +/-inf must match exactly
        if math.isnan(e) and math.isnan(a):
            return
        if math.isinf(e) or math.isinf(a):
            if e != a:
                raise AssertionError(f"{path}: inf mismatch — expected {e}, got {a}")
            return
        if not math.isclose(e, a, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(
                f"{path}: float mismatch — expected {e!r}, got {a!r} "
                f"(atol={atol}, rtol={rtol})"
            )
        return

    if expected != actual:
        raise AssertionError(f"{path}: value mismatch — expected {expected!r}, got {actual!r}")


def assert_snapshot_match(
    name: str,
    data: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """Assert that ``data`` matches the snapshot at ``snapshots/<name>.json``.

    If ``LZG_REGEN_SNAPSHOTS`` is set, the snapshot is overwritten instead.
    """
    path = SNAPSHOT_DIR / f"{name}.json"
    sanitized = _to_json_value(data)

    if _regen_mode():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n")
        return

    if not path.exists():
        raise AssertionError(
            f"Snapshot not found: {path}\n"
            f"Run with LZG_REGEN_SNAPSHOTS=1 to create it."
        )

    expected = _from_json_value(json.loads(path.read_text()))
    actual = _from_json_value(sanitized)
    _compare(name, expected, actual, atol, rtol)
