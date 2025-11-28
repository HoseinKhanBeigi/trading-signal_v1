"""
Main Crypto Velocity Tracker - Orchestrates all components
"""

import time
import threading
from typing import Dict, List
from price_tracker import PriceTracker
from indicators import TechnicalIndicators
from price_prediction import PricePredictor
from signal_generator import SignalGenerator
from websocket_handler import WebSocketHandler
from alert_handler import AlertHandler
from data_collector import DataCollector
from config import COLLECT_DATA, COLLECT_PRICES, COLLECT_FEATURES, COLLECT_SIGNALS, DEBUG_RUNTIME


class CryptoVelocityTracker:
    """Main class that orchestrates all components"""
    
    def __init__(self, symbols: List[str], timeframes: List[int], 
                 weak_threshold: float = 0.05, strong_threshold: float = 0.15):
        """
        Initialize the tracker
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., ['BTC', 'ETH', 'DOGE', 'XRP'])
            timeframes: List of timeframes in minutes (e.g., [1, 3, 15])
            weak_threshold: Velocity threshold for weak signal (default: 0.05 %/min)
            strong_threshold: Velocity threshold for strong signal (default: 0.15 %/min)
        """
        self.symbols = [s.upper() for s in symbols]
        self.timeframes = timeframes
        self.current_prices = {symbol: None for symbol in self.symbols}
        self.running = False
        self.lock = threading.Lock()
        
        # Signal tracking
        self.weak_threshold = weak_threshold
        self.strong_threshold = strong_threshold
        self.last_signals = {symbol: {tf: None for tf in timeframes} for symbol in self.symbols}
        
        # Initialize components
        self.price_tracker = PriceTracker(symbols, timeframes)
        self.indicators = TechnicalIndicators(self.price_tracker)
        self.predictor = PricePredictor(weak_threshold, strong_threshold, use_ai=True)
        self.signal_generator = SignalGenerator(
            self.indicators, self.predictor, weak_threshold, strong_threshold)
        self.alert_handler = AlertHandler()
        # Only initialize data collector if data collection is enabled
        self.data_collector = DataCollector() if COLLECT_DATA else None
        
        # WebSocket handler
        self.ws_handler = WebSocketHandler(symbols, self._on_price_update)
    
    def _on_price_update(self, symbol: str, price: float, timestamp: float):
        """Callback when price updates from WebSocket"""
        with self.lock:
            if symbol in self.symbols:
                self.current_prices[symbol] = price
                self.price_tracker.update_price_history(symbol, price, timestamp)
                # Save price data to file (if enabled)
                if self.data_collector and COLLECT_PRICES:
                    self.data_collector.save_price_data(symbol, price, timestamp)
    
    def get_all_velocities(self) -> Dict:
        """
        Get velocity data for all symbols and timeframes
        
        Returns:
            Dictionary with velocity data
        """
        results = {}
        
        for symbol in self.symbols:
            results[symbol] = {}
            for timeframe in self.timeframes:
                velocity, direction, change_pct, oldest_price, newest_price = \
                    self.price_tracker.calculate_velocity(symbol, timeframe)
                
                # Handle None values
                if velocity is None:
                    velocity = 0.0
                if change_pct is None:
                    change_pct = 0.0
                if newest_price is None:
                    newest_price = self.current_prices.get(symbol, 0.0)
                
                # Generate signal with all parameters
                current_price = self.current_prices.get(symbol, newest_price)
                if current_price is None or current_price == 0:
                    continue
                
                signal_type, signal_strength, signal_details = self.signal_generator.generate_signal(
                    symbol, timeframe, velocity, change_pct, current_price)
                
                results[symbol][f"{timeframe}min"] = {
                    "velocity": velocity,
                    "direction": direction,
                    "change_percent": change_pct,
                    "oldest_price": oldest_price,
                    "newest_price": newest_price,
                    "data_points": len(self.price_tracker.price_history[symbol][timeframe]),
                    "signal": signal_type,
                    "signal_strength": signal_strength,
                    "signal_details": signal_details
                }
        
        return results
    
    def check_and_alert(self, symbol: str, timeframe: int, velocity: float, 
                       change_pct: float, current_price: float):
        """
        Check for signals and alert if new signal detected
        """
        if velocity is None:
            return
        
        # Handle None change_pct
        if change_pct is None:
            change_pct = 0.0
        
        signal_type, signal_strength, signal_details = self.signal_generator.generate_signal(
            symbol, timeframe, velocity, change_pct, current_price)
        last_signal = self.last_signals[symbol][timeframe]
        
        # Don't alert for HOLD/NEUTRAL signals
        if signal_type == "HOLD ⏸️" or signal_strength == "NEUTRAL":
            # Update last signal but don't alert
            self.last_signals[symbol][timeframe] = signal_type
            return
        
        # Don't alert for WEAK signals - only STRONG signals
        if signal_strength == "WEAK":
            # Update last signal but don't alert
            self.last_signals[symbol][timeframe] = signal_type
            return
        
        # Only alert for VERY STRONG signals (STRONG BUY or STRONG SELL)
        if signal_strength == "VERY STRONG":
            self.alert_handler.alert_signal(
                symbol, timeframe, signal_type, signal_strength, 
                velocity, change_pct, current_price, signal_details)
            self.last_signals[symbol][timeframe] = signal_type
            
            # Save signal data (if enabled)
            if self.data_collector and COLLECT_SIGNALS:
                self.data_collector.save_signal(symbol, {
                    "signal_type": signal_type,
                    "signal_strength": signal_strength,
                    "velocity": velocity,
                    "change_pct": change_pct,
                    "price": current_price,
                    "timeframe": timeframe,
                    **signal_details
                })
    
    def _signal_check_loop(self, check_interval: float = 1.0):
        """Check for signals in background - alerts only (plus optional debug output)"""
        while self.running:
            try:
                with self.lock:
                    results = self.get_all_velocities()
                
                # Check for signals and alert
                for symbol in self.symbols:
                    current_price = self.current_prices.get(symbol)
                    if current_price is None:
                        continue
                    
                    for timeframe in self.timeframes:
                        tf_key = f"{timeframe}min"
                        data = results[symbol][tf_key]
                        
                        velocity = data.get("velocity")
                        if velocity is not None:
                            change_pct = data.get("change_percent", 0.0)
                            # Optional debug line for 3min timeframe
                            if DEBUG_RUNTIME and timeframe == 3:
                                details = data.get("signal_details", {}) or {}
                                pred = details.get("predicted_change_pct", 0.0)
                                conf = details.get("prediction_confidence", 0.0)
                                sig = data.get("signal", "")
                                strength = data.get("signal_strength", "")
                                # Highlight STRONG SELL in red-like formatting
                                if "STRONG SELL" in sig:
                                    print(f"[{symbol} {timeframe}m] ⚠️  STRONG SELL 🔻 "
                                          f"vel={velocity:+.4f}%/min "
                                          f"chg={change_pct:+.3f}% "
                                          f"pred={pred:+.3f}% "
                                          f"conf={conf*100:.1f}%")
                                elif "SELL" in sig:
                                    print(f"[{symbol} {timeframe}m] SELL 📉 "
                                          f"vel={velocity:+.4f}%/min "
                                          f"chg={change_pct:+.3f}% "
                                          f"pred={pred:+.3f}% "
                                          f"conf={conf*100:.1f}%")
                                else:
                                    print(f"[{symbol} {timeframe}m] "
                                          f"vel={velocity:+.4f}%/min "
                                          f"chg={change_pct:+.3f}% "
                                          f"sig={sig} "
                                          f"pred={pred:+.3f}% "
                                          f"conf={conf*100:.1f}%")
                            # Check and alert for 3min timeframe only
                            if timeframe == 3:
                                self.check_and_alert(symbol, timeframe, velocity, change_pct, current_price)
                
                time.sleep(check_interval)
            except Exception:
                # Silently ignore loop errors
                pass
    
    def run_continuous(self, check_interval: float = 1.0):
        """
        Run continuous tracking via WebSocket - only shows signal alerts
        
        Args:
            check_interval: Signal check interval in seconds (default: 1.0 second)
        """
        self.running = True
        
        # Start signal checking thread (only prints alerts, no dashboard)
        signal_thread = threading.Thread(target=self._signal_check_loop, args=(check_interval,), daemon=True)
        signal_thread.start()
        
        try:
            # Connect to WebSocket (blocking)
            self.ws_handler.connect()
            
        except KeyboardInterrupt:
            self.running = False
            self.ws_handler.close()
        except Exception:
            self.running = False
