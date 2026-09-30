"""Operator fare clearing.

The first close (``initial_clearing_id`` absent) freezes the half-open
settlement window (on the trips' tap-out time ``ended_at``) together with the
operator allocation rules, reads the trips currently inside the window, splits
every trip amount by the station-pair rule, and appends one immutable ledger
row per (trip, operator). Rows are only ever inserted.

Tap corrections later can change trip amounts, stations or pairings. A
follow-up batch carries a fresh ``clearing_id`` and references the first close
via ``initial_clearing_id``; it inherits the frozen window and rules,
re-splits every currently in-window trip, compares each operator share with
the cumulative amounts already posted by every batch of the chain, and
appends *only* the signed difference rows:

* amount grew or shrank      -> positive / negative top-up per operator;
* stations changed           -> the old allocation is fully reversed and the
                                 new rule's shares are appended;
* trip disappeared           -> every historical row for it is reversed;
* no change                  -> no row is appended.

Re-posting the same adjustment (same ``clearing_id``, same canonical payload)
replays the stored batch, so a retried adjustment can never be booked twice.

Indivisible-cent splitting uses the largest-remainder rule: each operator gets
``floor(amount * share / 10000)``; the leftover cents go one each to the
operators with the largest fractional remainders, ties broken by operator id
ascending, so the row totals always equal the trip amount exactly.
"""
from __future__ import annotations

import hashlib
import json

from .audit import canonical_json
from .settlement import iso_str


def normalise_rules(rules) -> list[dict]:
    """Validate and canonicalise the request rules: deterministic order, no
    duplicate station pairs / operators, every pair's shares total exactly
    10000 basis points. Raises ``ValueError`` with an HTTP-422 detail."""
    normalised = []
    seen_pairs: set[tuple[str, str]] = set()
    for r in rules or []:
        pair = (r.from_station, r.to_station)
        if pair in seen_pairs:
            raise ValueError(
                f"duplicate rule for station pair {r.from_station} -> {r.to_station}"
            )
        seen_pairs.add(pair)
        seen_ops: set[str] = set()
        shares: list[dict] = []
        for s in r.shares:
            if s.operator_id in seen_ops:
                raise ValueError(
                    f"rule {r.from_station} -> {r.to_station} lists operator"
                    f" '{s.operator_id}' more than once"
                )
            seen_ops.add(s.operator_id)
            shares.append(
                {"operator_id": s.operator_id, "basis_points": s.basis_points}
            )
        total = sum(s["basis_points"] for s in shares)
        if total != 10000:
            raise ValueError(
                f"rule {r.from_station} -> {r.to_station} shares total {total},"
                " must total exactly 10000 basis points"
            )
        shares.sort(key=lambda s: s["operator_id"])
        normalised.append(
            {
                "from_station": r.from_station,
                "to_station": r.to_station,
                "shares": shares,
            }
        )
    normalised.sort(key=lambda r: (r["from_station"], r["to_station"]))
    return normalised


def canonical_initial_payload(
    clearing_id: str, window_start, window_end, rules: list[dict]
) -> dict:
    """Canonical first-close request: timestamps normalised to UTC, rules
    sorted by station pair and operator id, so semantically identical
    submissions hash identically regardless of array order or offset notation."""
    return {
        "clearing_id": clearing_id,
        "window_start": iso_str(window_start),
        "window_end": iso_str(window_end),
        "rules": rules,
        "initial_clearing_id": None,
    }


def canonical_adjustment_payload(
    clearing_id: str, initial_clearing_id: str
) -> dict:
    """An adjustment's identity is its own id plus the first close it attaches
    to; the window and rules are inherited from the frozen close and never
    part of the request."""
    return {
        "clearing_id": clearing_id,
        "window_start": None,
        "window_end": None,
        "rules": None,
        "initial_clearing_id": initial_clearing_id,
    }


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def split_amount(amount_cents: int, shares: list[dict]) -> list[tuple[str, int]]:
    """Largest-remainder allocation of ``amount_cents`` into integer cents.

    Each share receives ``floor(amount * basis_points / 10000)``; the leftover
    cents are handed out one each in descending fractional-remainder order,
    ties broken by operator id ascending. The returned pieces always sum to
    ``amount_cents`` exactly.
    """
    total_bps = sum(s["basis_points"] for s in shares)  # always 10000 after validation
    floors = {
        s["operator_id"]: amount_cents * s["basis_points"] // total_bps
        for s in shares
    }
    allocated = sum(floors.values())
    leftover = amount_cents - allocated

    def remainder_key(s: dict) -> tuple:
        exact = amount_cents * s["basis_points"]
        remainder = exact % total_bps  # numerator of the fractional cent
        # Larger fraction first; equal fractions resolve to the smaller id.
        return (-remainder, s["operator_id"])

    ranked = sorted(shares, key=remainder_key)
    for s in ranked[:leftover]:
        floors[s["operator_id"]] += 1
    return sorted(floors.items())


def _rule_map(rules: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["from_station"], r["to_station"]): r for r in rules}


def find_missing_rule_pairs(trips: list[dict], rules: list[dict]) -> list[tuple[str, str]]:
    """Station pairs present among in-window trips but absent from the rules."""
    rule_map = _rule_map(rules)
    missing = {
        (t["from_station"], t["to_station"])
        for t in trips
        if (t["from_station"], t["to_station"]) not in rule_map
    }
    return sorted(missing)


def build_initial_entries(
    trips: list[dict], rules: list[dict], created_at: str
) -> list[dict]:
    """Ledger rows for the first close: the full amount per operator. The
    signed posting (``delta_cents``) is the allocation itself and the running
    net (``cumulative_amount_cents``) starts at that value."""
    rule_map = _rule_map(rules)
    entries: list[dict] = []
    for t in trips:
        pieces = split_amount(
            t["amount_cents"],
            rule_map[(t["from_station"], t["to_station"])]["shares"],
        )
        for operator_id, amount in pieces:
            entries.append(
                {
                    "trip_id": t["trip_id"],
                    "from_station": t["from_station"],
                    "to_station": t["to_station"],
                    "operator_id": operator_id,
                    "delta_cents": amount,
                    "current_amount_cents": t["amount_cents"],
                    "cumulative_amount_cents": amount,
                    "created_at": created_at,
                }
            )
    entries.sort(key=lambda e: (e["trip_id"], e["operator_id"]))
    for seq, e in enumerate(entries):
        e["seq"] = seq
    return entries


def build_adjustment_entries(
    trips: list[dict],
    rules: list[dict],
    previous_entries: list[dict],
    created_at: str,
) -> list[dict]:
    """Signed-difference rows for a follow-up batch.

    ``target = split(current amount)`` per in-window trip; for every
    (trip, operator) the posted amount is the cumulative sum of that
    operator's rows across every earlier batch of the chain. Only non-zero
    ``target - posted`` differences are appended, including full reversals
    for operators dropped from the trip's current rule and for trips that no
    longer exist inside the window.
    """
    rule_map = _rule_map(rules)
    current = {t["trip_id"]: t for t in trips}

    # Everything previously booked for these trips, keyed for accumulation.
    posted: dict[tuple[str, str], int] = {}
    pairs: dict[str, tuple[str, str]] = {}
    for e in previous_entries:
        key = (e["trip_id"], e["operator_id"])
        posted[key] = posted.get(key, 0) + e["delta_cents"]
        pairs.setdefault(
            e["trip_id"], (e["from_station"], e["to_station"])
        )

    targets: dict[tuple[str, str], int] = {}
    for t in trips:
        pieces = split_amount(
            t["amount_cents"],
            rule_map[(t["from_station"], t["to_station"])]["shares"],
        )
        pairs[t["trip_id"]] = (t["from_station"], t["to_station"])
        for operator_id, amount in pieces:
            targets[(t["trip_id"], operator_id)] = amount

    keys = set(posted) | set(targets)
    rows: list[dict] = []
    for trip_id, operator_id in sorted(keys):
        delta = targets.get((trip_id, operator_id), 0) - posted.get(
            (trip_id, operator_id), 0
        )
        if delta == 0:
            continue
        from_station, to_station = pairs[trip_id]
        cur = current.get(trip_id)
        rows.append(
            {
                "trip_id": trip_id,
                "from_station": from_station,
                "to_station": to_station,
                "operator_id": operator_id,
                "delta_cents": delta,
                "current_amount_cents": cur["amount_cents"] if cur is not None else 0,
                "cumulative_amount_cents": posted.get((trip_id, operator_id), 0) + delta,
                "created_at": created_at,
            }
        )
    rows.sort(key=lambda e: (e["trip_id"], e["operator_id"]))
    for seq, e in enumerate(rows):
        e["seq"] = seq
    return rows
