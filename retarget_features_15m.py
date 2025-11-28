"""
Retarget existing feature files to 15-minute (or N-minute) price-change targets
using locally stored price data.

Usage:
    python3 retarget_features_15m.py

It will:
    - For each symbol in config.SYMBOLS
    - Read data/prices/{symbol}_prices.jsonl
    - Read data/features/{symbol}_features.jsonl (old 5-min targets)
    - Compute new targets based on PREDICTION_MINUTES_AHEAD minutes ahead
    - Write data/features/{symbol}_features_15m.jsonl
"""

import json
import os
from datetime import datetime, timedelta
from bisect import bisect_right
from typing import List, Tuple

from config import SYMBOLS, PREDICTION_MINUTES_AHEAD


def load_price_series(symbol: str) -> List[Tuple[float, float]]:
    """Load (timestamp, price) list from prices JSONL."""
    path = os.path.join("data", "prices", f"{symbol.lower()}_prices.jsonl")
    if not os.path.exists(path):
        return []

    series: List[Tuple[float, float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ts = float(data.get("timestamp"))
                price = float(data.get("price"))
                series.append((ts, price))
            except Exception:
                continue

    series.sort(key=lambda x: x[0])
    return series


def build_timestamp_index(series: List[Tuple[float, float]]):
    """Return separate lists of timestamps and prices for bisect."""
    timestamps = [ts for ts, _ in series]
    prices = [p for _, p in series]
    return timestamps, prices


def find_price_at_or_before(
    timestamps: List[float], prices: List[float], target_ts: float
) -> float:
    """
    Find last price with timestamp <= target_ts.
    Returns None if not found.
    """
    idx = bisect_right(timestamps, target_ts) - 1
    if idx < 0 or idx >= len(prices):
        return None
    return prices[idx]


def retarget_symbol(symbol: str, horizon_minutes: int):
    """Retarget one symbol's feature file to new horizon."""
    price_series = load_price_series(symbol)
    if not price_series:
        return

    timestamps, prices = build_timestamp_index(price_series)

    in_path = os.path.join("data", "features", f"{symbol.lower()}_features.jsonl")
    out_path = os.path.join(
        "data", "features", f"{symbol.lower()}_features_15m.jsonl"
    )

    if not os.path.exists(in_path):
        return

    horizon_seconds = horizon_minutes * 60

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            ts_str = data.get("timestamp")
            try:
                base_dt = datetime.fromisoformat(ts_str)
            except Exception:
                continue

            t0 = base_dt.timestamp()
            t1 = t0 + horizon_seconds

            p0 = find_price_at_or_before(timestamps, prices, t0)
            p1 = find_price_at_or_before(timestamps, prices, t1)

            if p0 is None or p1 is None or p0 <= 0:
                continue

            target_change = ((p1 - p0) / p0) * 100.0
            data["target"] = target_change

            try:
                fout.write(json.dumps(data) + "\n")
            except Exception:
                continue


def main():
    horizon = int(PREDICTION_MINUTES_AHEAD)
    for symbol in SYMBOLS:
        retarget_symbol(symbol, horizon)


if __name__ == "__main__":
    main()


