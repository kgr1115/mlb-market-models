"""
Higher-level modeling package.

Right now this holds the closing-line predictor, which reframes the
betting problem from "who wins the game" to "where will this line
close?". See closing_line.py for the theory.
"""
from .closing_line import (
    ClosingLinePrediction,
    predict_closing_line_ml,
    predict_closing_line_total,
    predict_closing_line_rl,
    predicted_closing_by_event,
)
from .opening_lines import (
    OpeningLineStore, record_opener, get_opener, get_all_openers_today,
)

__all__ = [
    "ClosingLinePrediction",
    "predict_closing_line_ml", "predict_closing_line_total", "predict_closing_line_rl",
    "predicted_closing_by_event",
    "OpeningLineStore", "record_opener", "get_opener", "get_all_openers_today",
]
