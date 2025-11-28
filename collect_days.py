#!/usr/bin/env python3
"""
Script to automatically collect historical data day by day
Usage: python3 collect_days.py [start_day] [end_day]
Example: python3 collect_days.py 0 120  (collects days 0 to 120)
"""

import sys
import subprocess
import time
from historical_data_collector import collect_single_day
from config import SYMBOLS

def main():
    """Collect data for a range of days (silent)."""
    # Get start and end day from command line arguments
    if len(sys.argv) >= 3:
        start_day = int(sys.argv[1])
        end_day = int(sys.argv[2])
    elif len(sys.argv) == 2:
        start_day = int(sys.argv[1])
        end_day = 120
    else:
        start_day = 0
        end_day = 120
    
    for day in range(start_day, end_day + 1):
        try:
            collect_single_day(SYMBOLS, days_ago=day)
        except Exception:
            # Continue to next day silently
            pass
        
        if day < end_day:
            time.sleep(2)


if __name__ == "__main__":
    main()

