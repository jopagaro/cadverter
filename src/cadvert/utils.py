"""Shared utilities: hashing, rounding, unit formatting."""

import hashlib
import math
from pathlib import Path


DISPLAY_PRECISION = 3  # decimal places in output


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value: float, prec: int = DISPLAY_PRECISION) -> str:
    """Format a float to fixed decimal places, stripping unnecessary trailing zeros."""
    return f"{value:.{prec}f}"


def fmt_pt(pt: tuple[float, float, float], prec: int = DISPLAY_PRECISION) -> str:
    return f"({fmt(pt[0], prec)}, {fmt(pt[1], prec)}, {fmt(pt[2], prec)})"


def fmt_vec(v: tuple[float, float, float], prec: int = DISPLAY_PRECISION) -> str:
    return f"({fmt(v[0], prec)}, {fmt(v[1], prec)}, {fmt(v[2], prec)})"


def angle_between_normals(n1: tuple, n2: tuple) -> float:
    """Return angle in degrees between two unit normal vectors."""
    dot = sum(a * b for a, b in zip(n1, n2))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = math.sqrt(sum(x * x for x in v))
    if mag < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def cross(a: tuple, b: tuple) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
