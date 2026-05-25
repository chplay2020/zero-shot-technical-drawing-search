from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Candidate:
    """Candidate bounding box with scale and rotation metadata."""

    x: int
    y: int
    w: int
    h: int
    scale: float
    rotation: float


def generate_candidates(
    image_shape: Tuple[int, int],
    query_shape: Tuple[int, int],
    scales: Iterable[float],
    rotations: Iterable[float],
    max_candidates: int,
) -> List[Candidate]:
    """Generate sliding-window candidates across scales and rotations."""
    h, w = image_shape
    qh, qw = query_shape
    candidates: List[Candidate] = []
    for scale in scales:
        win_w = max(4, int(round(qw * scale)))
        win_h = max(4, int(round(qh * scale)))
        if win_w >= w or win_h >= h:
            continue
        step = max(4, min(win_w, win_h) // 4)
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                for rot in rotations:
                    candidates.append(Candidate(x, y, win_w, win_h, scale, rot))
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates
