"""
Adaptive labeling strategy: uncertainty sampling via subject ability.

Score each candidate by how uncertain our current prediction is (p near 0.5
is most valuable to label). Uses only subject theta — no model downloads needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_HERE = Path(__file__).resolve().parent

with open(_HERE / "subject_theta.json") as _f:
    _THETA_DATA = json.load(_f)

_THETA_BY_ID: dict[str, float] = _THETA_DATA["by_id_or_name"]
_THETA_BY_CONTENT: dict[str, float] = _THETA_DATA["by_content"]
_MEAN_THETA: float = _THETA_DATA["mean_theta"]
_TRAINING_PASS_RATE = 0.664
_DEFAULT_ALPHA: float = math.log(_TRAINING_PASS_RATE / (1.0 - _TRAINING_PASS_RATE)) - _MEAN_THETA


def _get_theta(subject_content: str) -> float:
    if subject_content in _THETA_BY_CONTENT:
        return _THETA_BY_CONTENT[subject_content]
    first_line = subject_content.split("\n")[0] if subject_content else ""
    name = first_line.replace("Name:", "").strip()
    return _THETA_BY_ID.get(name, _MEAN_THETA)


def acquisition_function(input: dict) -> float:
    """
    Score by prediction uncertainty: p closest to 0.5 is most informative.
    Labels on uncertain predictions best calibrate the benchmark difficulty offset.
    """
    subject_content: str = input.get("subject_content", "")
    theta = _get_theta(subject_content)
    p = 1.0 / (1.0 + math.exp(-(theta + _DEFAULT_ALPHA)))
    # Higher score = more desired. Uncertainty is maximized at p=0.5.
    return float(1.0 - abs(2.0 * p - 1.0))
