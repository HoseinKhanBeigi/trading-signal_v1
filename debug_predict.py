"""
Quick debug script to compare AI prediction vs actual 15-minute target.

Usage (from project root):
    python3 debug_predict.py            # defaults to BTC
    python3 debug_predict.py ETH        # or any symbol in your data
"""

import json
import os
import sys
from collections import deque

import numpy as np

from ai_predictor import AIPredictor


def load_last_features(symbol: str, count: int = 60):
    """
    Load the last `count` feature vectors for a symbol from *_features_15m.jsonl.

    Returns:
        (feature_history, last_target)
        - feature_history: list of numpy arrays (length <= count)
        - last_target: float (target % change from last line) or None
    """
    path = os.path.join(
        "data", "features", f"{symbol.lower()}_features_15m.jsonl"
    )
    if not os.path.exists(path):
        print(f"No feature file found at: {path}")
        return [], None

    buf = deque(maxlen=count)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                buf.append(data)
            except Exception:
                continue

    if not buf:
        print(f"No valid feature lines found in: {path}")
        return [], None

    feature_history = []
    last_target = None
    for entry in buf:
        feats = entry.get("features")
        if isinstance(feats, list) and len(feats) > 0:
            feature_history.append(np.array(feats, dtype=np.float32))
        last_target = entry.get("target", None)

    return feature_history, last_target


def main():
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "BTC"
    print(f"Debugging prediction for symbol: {symbol}")

    feature_history, last_target = load_last_features(symbol, count=60)
    if len(feature_history) < 60:
        print(f"Not enough feature history (got {len(feature_history)}, need 60).")
        return

    predictor = AIPredictor()
    if not predictor.is_trained:
        print("Warning: model is not marked as trained; predictions may be fallback.")

    # We only care about predicted % change, so we can set current_price = 1.0
    current_price = 1.0
    result = predictor.predict(feature_history, current_price)

    pred_pct = result.get("predicted_change_pct", 0.0)
    conf = result.get("confidence", 0.0)

    print("\n=== Debug Prediction ===")
    print(f"Method        : {result.get('method')}")
    print(f"Predicted Δ%  : {pred_pct:+.4f}%")
    if last_target is not None:
        print(f"Actual target : {last_target:+.4f}%")
        diff = pred_pct - float(last_target)
        print(f"Error (pred-actual): {diff:+.4f}%")
    else:
        print("Actual target : N/A (no target field found)")
    print(f"Confidence    : {conf*100:.1f}%")
    print("========================\n")


if __name__ == "__main__":
    main()


