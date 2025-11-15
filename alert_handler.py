"""
Alert handler for trading signals
"""

from datetime import datetime
from typing import Dict
import os
import subprocess
import platform


class AlertHandler:
    """Handle signal alerts"""
    
    def __init__(self):
        """Initialize alert handler"""
        self.signal_count = 0
    
    def alert_signal(self, symbol: str, timeframe: int, signal_type: str, 
                    signal_strength: str, velocity: float, change_pct: float, 
                    price: float, signal_details: Dict):
        """
        Print alert signal with detailed indicators
        """
        self.signal_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Visual alert with colors and emphasis
        print("\n" + "="*80)
        print(f"🔔 CRYPTO SIGNAL #{self.signal_count} - {timestamp}")
        print("="*80)
        
        if "STRONG" in signal_type:
            # Strong signals get extra emphasis
            print(f"⚠️  {signal_type} - {signal_strength}")
            
            # Trigger system alert with sound for BOTH STRONG BUY and STRONG SELL
            if "STRONG BUY" in signal_type:
                self._trigger_system_alert(symbol, signal_type, price, signal_details['predicted_change_pct'], is_buy=True)
            elif "STRONG SELL" in signal_type:
                self._trigger_system_alert(symbol, signal_type, price, signal_details['predicted_change_pct'], is_buy=False)
        else:
            print(f"   {signal_type} - {signal_strength}")
        
        print(f"\n   Symbol: {symbol}/USDT")
        print(f"   Timeframe: {timeframe}min")
        print(f"   Current Price: ${price:.4f}")
        print(f"\n   📊 ALL TRADING PARAMETERS:")
        print(f"   • Velocity: {velocity:+.4f} %/min")
        print(f"   • Momentum: {signal_details['momentum']:+.4f} (acceleration)")
        print(f"   • Trend Strength: {signal_details['trend_strength']*100:.1f}% (consistency)")
        print(f"   • RSI: {signal_details['rsi']:.2f} ({'Overbought' if signal_details['rsi'] > 70 else 'Oversold' if signal_details['rsi'] < 30 else 'Neutral'})")
        print(f"   • EMA: ${signal_details['ema']:.4f} (Price {signal_details['ema_position']})")
        print(f"   • Support: ${signal_details['support']:.4f} | Resistance: ${signal_details['resistance']:.4f}")
        print(f"   • Price Position: {signal_details['price_position']}")
        print(f"   • Current Change: {change_pct:+.3f}%")
        print(f"\n   🔮 PRICE PREDICTION (5 min ahead):")
        print(f"   • Predicted Price: ${signal_details['predicted_price']:.4f}")
        print(f"   • Predicted Change: {signal_details['predicted_change_pct']:+.3f}%")
        print(f"   • Confidence: {signal_details['prediction_confidence']*100:.1f}%")
        print("="*80 + "\n")
    
    def _trigger_system_alert(self, symbol: str, signal_type: str, price: float, predicted_change: float, is_buy: bool = True):
        """
        Trigger system notification with sound for STRONG BUY and STRONG SELL signals
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            signal_type: Signal type (e.g., 'STRONG BUY 🚀' or 'STRONG SELL 🔻')
            price: Current price
            predicted_change: Predicted price change percentage
            is_buy: True for BUY signals, False for SELL signals
        """
        try:
            if platform.system() == "Darwin":  # macOS
                # Create notification message
                title = f"{signal_type} - {symbol}/USDT"
                message = f"Price: ${price:.4f} | Predicted: {predicted_change:+.2f}%"
                
                # Use different sounds for BUY vs SELL
                if is_buy:
                    sound_name = "Glass"  # Pleasant sound for BUY
                    sound_file = "/System/Library/Sounds/Glass.aiff"
                else:
                    sound_name = "Basso"  # Warning sound for SELL
                    sound_file = "/System/Library/Sounds/Basso.aiff"
                
                # Use osascript to show notification with sound
                script = f'''
                display notification "{message}" with title "{title}" sound name "{sound_name}"
                '''
                
                subprocess.run(
                    ["osascript", "-e", script],
                    check=False,
                    capture_output=True
                )
                
                # Also play a system sound file directly
                os.system(f'afplay {sound_file} 2>/dev/null || echo -e "\\a"')
                
        except Exception as e:
            # Silently fail if notification fails (don't interrupt main flow)
            pass

