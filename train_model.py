"""
Training script for PatchTST AI model
Run this after collecting historical data to train the model
"""

from ai_predictor import AIPredictor
from data_collector import DataCollector
import numpy as np
from typing import List, Tuple
import time


def collect_training_data_from_live_system(tracker, duration_minutes: int = 60) -> List[Tuple]:
    """Collect training data from live system."""
    training_data = []
    start_time = time.time()
    
    while time.time() - start_time < duration_minutes * 60:
        try:
            # Get current velocities and indicators
            results = tracker.get_all_velocities()
            
            for symbol in tracker.symbols:
                for timeframe in tracker.timeframes:
                    tf_key = f"{timeframe}min"
                    data = results[symbol].get(tf_key, {})
                    
                    if data.get("velocity") is not None:
                        # Get feature history
                        if hasattr(tracker.predictor, 'feature_history'):
                            feature_history = tracker.predictor.feature_history.get(symbol, [])
                            
                            if len(feature_history) >= 60:
                                # Get actual price change after 5 minutes
                                # This would need to be stored and retrieved later
                                # For now, use predicted change as target (will improve with real data)
                                current_price = tracker.current_prices.get(symbol, 0)
                                if current_price > 0:
                                    # Use velocity-based target (will be replaced with actual data)
                                    target_change = data.get("change_percent", 0.0)
                                    training_data.append((feature_history[-60:], target_change))
            
            time.sleep(10)  # Collect every 10 seconds
            
        except Exception:
            time.sleep(5)
    
    return training_data


def train_model_with_data(training_data: List[Tuple], epochs: int = 50):
    """Train the AI model with collected data."""
    if len(training_data) < 100:
        print(f"Not enough samples to train (got {len(training_data)}). Need >= 100.")
        return

    # Initialize predictor
    predictor = AIPredictor()
    
    print(f"Starting training on {len(training_data)} samples for {epochs} epochs...")
    predictor.train_model(training_data, epochs=epochs)
    print("Training run finished.")


if __name__ == "__main__":
    import sys
    from config import SYMBOLS

    # Determine mode from CLI (default to 'train')
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "train"

    if mode == "export":
        predictor = AIPredictor()
        predictor.export_to_coreml()
        sys.exit(0)

    # Train mode
    collector = DataCollector()
    all_training_data = []

    print("Loading training data from 15m feature files...")
    for symbol in SYMBOLS:
        training_data = collector.load_training_data(symbol, min_sequences=60)
        if training_data:
            all_training_data.extend(training_data)

    total = len(all_training_data)
    if total < 100:
        print(f"Total samples loaded: {total} (need at least 100). Aborting training.")
    else:
        print(f"Total samples loaded: {total}. Starting training...")
        predictor = AIPredictor()
        predictor.train_model(all_training_data, epochs=30)

