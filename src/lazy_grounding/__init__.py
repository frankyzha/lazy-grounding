"""Evaluation tools for lazy grounding in search agents."""

from lazy_grounding.metrics import MetricSummary, summarize_outcomes
from lazy_grounding.schemas import EvidenceRecord, Outcome, QuestionItem, Rewrite, SearchCandidate

__all__ = [
    "EvidenceRecord",
    "MetricSummary",
    "Outcome",
    "QuestionItem",
    "Rewrite",
    "SearchCandidate",
    "summarize_outcomes",
]

__version__ = "0.1.0"
