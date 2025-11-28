"""
Collect historical data from Binance API for faster training
This allows training the AI model immediately without waiting for real-time data
"""

import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict
import json
from data_collector import DataCollector
from price_tracker import PriceTracker
from indicators import TechnicalIndicators
from price_prediction import PricePredictor
from config import PREDICTION_MINUTES_AHEAD


class HistoricalDataCollector:
    """Collect historical price data and generate training features"""
    
    def __init__(self):
        """Initialize historical data collector"""
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.data_collector = DataCollector()
    
    def fetch_historical_klines(self, symbol: str, interval: str = "1m", 
                                limit: int = 1000, end_time: int = None) -> List[List]:
        """
        Fetch historical kline/candlestick data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 3m, 5m, etc.)
            limit: Number of klines to fetch (max 1000)
            end_time: End time in milliseconds (optional, for batch collection)
            
        Returns:
            List of klines [open_time, open, high, low, close, volume, ...]
        """
        try:
            url = f"{self.base_url}?symbol={symbol}USDT&interval={interval}&limit={limit}"
            if end_time:
                url += f"&endTime={end_time}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception:
            return []
    
    def process_historical_data(self, symbol: str, days_back: int = 7, 
                                end_time: int = None):
        """
        Process historical data and generate training features
        """
        # Fetch historical klines (1-minute intervals)
        # Each day = 1440 minutes, so for many days we need multiple requests
        # Binance API limit: 1000 klines per request
        total_minutes = days_back * 1440
        klines = []
        
        # Fetch in batches of 1000 (Binance limit)
        batches = (total_minutes // 1000) + 1
        current_end_time = end_time
        
        for i in range(batches):
            limit = min(1000, total_minutes - len(klines))
            if limit <= 0:
                break
            
            batch = self.fetch_historical_klines(symbol, "1m", limit, current_end_time)
            if batch:
                # Binance returns klines in reverse chronological order (newest first)
                # Keep batches in reverse chronological order (newest first)
                # We'll reverse the entire list at the end to get chronological order
                # Then process in reverse to save newest first
                klines.extend(batch)
                # Update end_time for next batch (go further back in time)
                # Use the oldest kline's open_time as the new end_time
                # Since batch is newest first, oldest is at the end
                if batch:
                    current_end_time = batch[-1][0] - 1  # Oldest kline time - 1ms
                time.sleep(0.2)  # Rate limiting
        
        if len(klines) < 100:
            return
        
        # Reverse klines to get chronological order (oldest first)
        # Price tracker needs chronological order for indicators to work correctly
        klines.reverse()
        
        # Filter klines to only include data from the target day if end_time is specified
        # This ensures we only get data for the specific day requested
        if end_time is not None:
            # Calculate start of the day from end_time
            end_datetime = datetime.fromtimestamp(end_time / 1000)
            start_of_day = end_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            start_timestamp = int(start_of_day.timestamp() * 1000)
            
            # Filter klines to only include those from the target day
            original_count = len(klines)
            klines = [k for k in klines if start_timestamp <= k[0] <= end_time]
        
        # Process klines into prices
        prices = []
        timestamps = []
        
        for kline in klines:
            # kline format: [open_time, open, high, low, close, volume, ...]
            timestamp = kline[0] / 1000  # Convert milliseconds to seconds
            close_price = float(kline[4])  # Close price
            
            prices.append(close_price)
            timestamps.append(timestamp)
        
        # Create price tracker for this symbol
        price_tracker = PriceTracker([symbol], [3])  # 3-minute timeframe
        
        # Update price tracker with historical data in chronological order (oldest to newest)
        # This is required for indicators to calculate correctly
        for i in range(0, len(prices)):
            # Update price tracker with each price point sequentially
            price_tracker.update_price_history(symbol, prices[i], timestamps[i])
        
        # Initialize indicators
        indicators = TechnicalIndicators(price_tracker)
        predictor = PricePredictor(0.05, 0.15, use_ai=False)  # Don't need AI for data collection
        
        # Generate features and targets
        # Collect all samples first, then save in reverse order (newest first)
        samples_to_save = []  # List of (features, target, timestamp) tuples
        skipped_count = 0
        error_counts = {
            'velocity': 0,
            'ema': 0,
            'indicators': 0,
            'other': 0
        }
        
        # Now process data starting from index where we have enough history
        # Need at least 60 data points for indicators, plus horizon for target
        # Since prices is in chronological order (oldest first), prices[0] = oldest, prices[-1] = newest
        # We process in FORWARD order (oldest to newest) to collect all data
        start_idx = max(60, 20)  # Start after we have some history
        # Process from oldest to newest (forward through array: start to end)
        horizon = max(1, int(PREDICTION_MINUTES_AHEAD))
        for i in range(start_idx, len(prices) - horizon):
            try:
                # Get current price and time
                current_price = prices[i]
                current_time = timestamps[i]
                
                # Calculate indicators
                velocity, _, change_pct, _, _ = price_tracker.calculate_velocity(symbol, 3)
                if velocity is None:
                    error_counts['velocity'] += 1
                    skipped_count += 1
                    continue
                
                momentum = indicators.calculate_momentum(symbol, 3)
                rsi = indicators.calculate_rsi(symbol, 3)
                trend_strength = indicators.calculate_trend_strength(symbol, 3)
                
                ema_result = indicators.calculate_ema(symbol, 3)
                if not isinstance(ema_result, tuple):
                    error_counts['ema'] += 1
                    skipped_count += 1
                    continue
                
                ema, ema_position = ema_result
                
                # Get all indicators with better error handling
                try:
                    macd = indicators.calculate_macd(symbol, 3)
                    stochastic = indicators.calculate_stochastic(symbol, 3)
                    bollinger = indicators.calculate_bollinger_bands(symbol, 3)
                    atr = indicators.calculate_atr(symbol, 3)
                    adx = indicators.calculate_adx(symbol, 3)
                except Exception as e:
                    error_counts['indicators'] += 1
                    skipped_count += 1
                    continue
                
                # Prepare features
                all_indicators = {
                    'velocity': velocity,
                    'momentum': momentum if momentum else 0.0,
                    'rsi': rsi if rsi else 50.0,
                    'trend_strength': trend_strength if trend_strength else 0.5,
                    'ema_position': ema_position,
                    'macd': macd,
                    'stochastic': stochastic,
                    'bollinger': bollinger,
                    'atr': atr,
                    'adx': adx,
                    'current_price': current_price
                }
                
                # Calculate target (actual price change N minutes later where N = PREDICTION_MINUTES_AHEAD)
                idx_future = i + horizon
                if idx_future < len(prices):
                    future_price = prices[idx_future]
                    target_change = ((future_price - current_price) / current_price) * 100
                else:
                    continue  # Skip if we can't calculate target
                
                # Prepare features for saving
                features = predictor._prepare_ai_features(all_indicators)
                
                # Create timestamp from actual price data time (rounded to minutes)
                price_dt = datetime.fromtimestamp(current_time)
                # Round to nearest minute (remove seconds and microseconds)
                price_timestamp = price_dt.replace(second=0, microsecond=0).isoformat()
                
                # Collect sample (don't save yet)
                samples_to_save.append((features, target_change, price_timestamp))
                
            except Exception as e:
                error_counts['other'] += 1
                skipped_count += 1
                continue
        
        # Check for existing timestamps to avoid duplicates
        existing_timestamps = set()
        filename = f"{self.data_collector.data_dir}/features/{symbol.lower()}_features.jsonl"
        try:
            import os
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                import json
                                data = json.loads(line.strip())
                                existing_timestamps.add(data.get('timestamp', ''))
                            except:
                                pass
        except Exception:
            # Ignore duplicate-scan errors silently
            pass
        
        # Now save all samples in REVERSE order (newest first, going backwards)
        # This gives us: today → yesterday → ... (newest to oldest)
        # samples_to_save is in chronological order (oldest first), so we reverse it
        training_samples = []
        reversed_samples = list(reversed(samples_to_save))
        skipped_duplicates = 0
        for features, target_change, price_timestamp in reversed_samples:
            # Skip if this timestamp already exists
            if price_timestamp in existing_timestamps:
                skipped_duplicates += 1
                continue
            
            # Save training sample (save as single feature vector, not sequence)
            # The model expects sequences of 60, but we'll save individual features
            # and the training script will create sequences
            self.data_collector.save_feature_sequence(
                symbol,
                features.tolist() if hasattr(features, 'tolist') else list(features),
                target_change,
                timestamp=price_timestamp  # Use actual price data timestamp
            )
            
            # Add to existing set to avoid duplicates within this batch
            existing_timestamps.add(price_timestamp)
            training_samples.append((features, target_change))
        
        return training_samples
    
    def collect_for_all_symbols(self, symbols: List[str], days_back: int = 7):
        """Collect historical data for all symbols"""
        total_samples = 0
        for symbol in symbols:
            samples = self.process_historical_data(symbol, days_back)
            if samples:
                total_samples += len(samples)
            time.sleep(1)  # Rate limiting between symbols
        
        return total_samples
    
    def collect_in_batches(self, symbols: List[str], batch_days: int = 7, 
                          num_batches: int = 30):
        """
        Collect historical data in batches, going further back each time
        
        Args:
            symbols: List of cryptocurrency symbols
            batch_days: Number of days per batch (default: 7)
            num_batches: Number of batches to collect (default: 30 = 210 days)
        """
        total_samples = 0
        
        # Process batches in FORWARD order: Batch 1 (newest) first, then Batch 2, etc.
        # Within each batch, we process in reverse (newest to oldest)
        # This gives us: Batch 1 newest → Batch 1 older → ... → Batch 1 oldest → Batch 2 newest → ...
        for batch_num in range(1, num_batches + 1):
            # Calculate end_time for this batch
            # Batch 1: endTime = now (most recent data) - collects last 7 days
            # Batch 2: endTime = now - 7 days - collects 7-14 days ago
            # Batch 3: endTime = now - 14 days - collects 14-21 days ago
            # etc.
            days_back_from_now = (batch_num - 1) * batch_days
            end_time = int((datetime.now() - timedelta(days=days_back_from_now)).timestamp() * 1000)
            
            end_date = datetime.fromtimestamp(end_time / 1000)
            
            batch_samples = 0
            for symbol in symbols:
                samples = self.process_historical_data(symbol, batch_days, end_time)
                if samples:
                    batch_samples += len(samples)
                    total_samples += len(samples)
                time.sleep(1)  # Rate limiting between symbols
            
            if batch_num < num_batches:
                time.sleep(2)

        return total_samples


def collect_single_day(symbols: List[str], days_ago: int = 0):
    """
    Collect data for a single day - SIMPLE: just get that day's data
    
    Args:
        symbols: List of cryptocurrency symbols
        days_ago: 0 = today, 1 = yesterday, 2 = day before yesterday, etc.
    """
    collector = HistoricalDataCollector()
    
    # Calculate the specific day we want
    target_date = datetime.now() - timedelta(days=days_ago)
    # Get start of that day (00:00:00)
    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    # Get end of that day (23:59:59) or now if it's today
    if days_ago == 0:
        end_of_day = datetime.now()
    else:
        end_of_day = start_of_day.replace(hour=23, minute=59, second=59)
    
    day_name = "today" if days_ago == 0 else f"{days_ago} day(s) ago"
    # Use end_time as the end of the day (in milliseconds)
    end_time = int(end_of_day.timestamp() * 1000)
    
    # Calculate how many minutes to fetch (from start of day to end of day)
    minutes_to_fetch = int((end_of_day - start_of_day).total_seconds() / 60)
    # But we'll use days_back=1 and end_time to get exactly that day
    
    total_samples = 0
    for symbol in symbols:
        # Fetch data ending at end_of_day, going back 1 day
        # But we'll limit it to just that day by using end_time
        samples = collector.process_historical_data(symbol, days_back=1, end_time=end_time)
        if samples:
            total_samples += len(samples)
        time.sleep(1)  # Rate limiting between symbols
    
    return total_samples


def main():
    """Main function to collect historical data"""
    import sys
    from config import SYMBOLS
    
    collector = HistoricalDataCollector()
    # Check if user wants to collect a single day
    if len(sys.argv) > 1:
        try:
            days_ago = int(sys.argv[1])
            collect_single_day(SYMBOLS, days_ago=days_ago)
            return
        except ValueError:
            return
    
    # For now, use batch mode by default
    use_batch_mode = True
    
    if use_batch_mode:
        # Batch mode: Collect 7 days at a time, going further back
        batch_days = 7  # 7 days per batch
        num_batches = 30  # 30 batches = 210 days (7 months)
        collector.collect_in_batches(SYMBOLS, batch_days=batch_days, num_batches=num_batches)
    else:
        # Single batch mode (original)
        days = 210  # 7 months
        collector.collect_for_all_symbols(SYMBOLS, days_back=days)
    
if __name__ == "__main__":
    main()

