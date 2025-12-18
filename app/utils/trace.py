from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional


@dataclass
class TraceRecord:
    request_id: str
    start_ts: float
    end_ts: Optional[float] = None
    spans: Dict[str, float] = None

    def __post_init__(self):
        if self.spans is None:
            self.spans = {}

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_ts is None:
            return None
        return (self.end_ts - self.start_ts) * 1000.0


def new_request_id() -> str:
    return str(uuid.uuid4())


@contextmanager
def trace_span(record: TraceRecord, name: str) -> Iterator[None]:
    """
    Simple timing span context manager.
    Usage:
        rec = TraceRecord(request_id=..., start_ts=time.time())
        with trace_span(rec, "router"):
            ...
    """
    t0 = time.time()
    try:
        yield
    finally:
        record.spans[name] = (time.time() - t0) * 1000.0  # ms


def start_trace(request_id: Optional[str] = None) -> TraceRecord:
    return TraceRecord(request_id=request_id or new_request_id(), start_ts=time.time())


def end_trace(record: TraceRecord) -> TraceRecord:
    record.end_ts = time.time()
    return record
