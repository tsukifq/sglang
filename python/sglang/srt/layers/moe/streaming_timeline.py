"""CPU-only identity checks for source-indexed streaming timeline metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def source_lane_records(
    lane_events: Sequence[Mapping[str, Any]], psums: Sequence[Any]
) -> list[tuple[int, Mapping[str, Any], Any]]:
    """Associate event records with source-indexed counts without truncation.

    Both lane and wave submitters must retain each source's identity. A wave's
    compute_group_id may be shared by several sources and is not a rank ID.
    Reject incomplete/misordered records instead of emitting plausible but
    incorrectly labelled measurements from a deferred collector.
    """
    if len(lane_events) != len(psums):
        raise ValueError(
            f"streaming timeline/count cardinality mismatch: "
            f"events={len(lane_events)} sources={len(psums)}"
        )
    records = []
    for source_rank, psum in enumerate(psums):
        events = lane_events[source_rank]
        identity = events.get("source_rank") if isinstance(events, Mapping) else None
        if type(identity) is not int or identity != source_rank:
            raise ValueError(
                f"streaming timeline source identity mismatch at slot "
                f"{source_rank}: observed={identity!r}"
            )
        records.append((source_rank, events, psum))
    return records
