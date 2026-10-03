from __future__ import annotations

from itertools import permutations

from .common import FIELDS


def greedy_sequence(field_scores: dict[str, float], budget: int, start: tuple[str, ...] = ("F1",)) -> tuple[str, ...]:
    selected = list(start)
    while len(selected) < budget:
        remaining = [f for f in FIELDS if f not in selected]
        selected.append(max(remaining, key=lambda f: field_scores[f]))
    return tuple(selected)


def nonmyopic_sequence_oracle(field_scores: dict[str, float], budget: int, start: tuple[str, ...] = ("F1",)) -> tuple[str, ...]:
    remaining = [f for f in FIELDS if f not in start]
    best = None
    for tail in permutations(remaining, max(0, budget - len(start))):
        seq = tuple(start) + tail
        score = sum(field_scores[f] for f in seq) / len(seq)
        if best is None or score > best[0]:
            best = (score, seq)
    return best[1] if best else tuple(start)

