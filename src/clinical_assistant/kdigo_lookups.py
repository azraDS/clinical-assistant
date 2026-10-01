"""Deterministic KDIGO lookups (GFR/albuminuria categories, risk heat map) — encoded as code,
not retrieved from a PDF figure, because that's exact and testable. See notebooks/README.md §6.
"""

import math

INF = math.inf

G_RANGES = {
    "G1": (90, INF),
    "G2": (60, 90),
    "G3a": (45, 60),
    "G3b": (30, 45),
    "G3": (30, 60),
    "G4": (15, 30),
    "G5": (0, 15),
}
A_RANGES = {"A1": (0, 30), "A2": (30, 300), "A3": (300, INF)}  # mg/g

RISK_HEATMAP = {
    "G1": {"A1": "low", "A2": "moderate", "A3": "high"},
    "G2": {"A1": "low", "A2": "moderate", "A3": "high"},
    "G3a": {"A1": "moderate", "A2": "high", "A3": "very high"},
    "G3b": {"A1": "high", "A2": "very high", "A3": "very high"},
    "G4": {"A1": "very high", "A2": "very high", "A3": "very high"},
    "G5": {"A1": "very high", "A2": "very high", "A3": "very high"},
}


def gfr_category(egfr: float | None) -> str | None:
    if egfr is None:
        return None
    for cat, (lo, hi) in G_RANGES.items():
        if cat != "G3" and lo <= egfr < hi:
            return cat
    return None


def acr_category(acr_mg_g: float | None) -> str | None:
    if acr_mg_g is None:
        return None
    for cat, (lo, hi) in A_RANGES.items():
        if lo <= acr_mg_g < hi:
            return cat
    return None
