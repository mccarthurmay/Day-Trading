import os
import time
from datetime import datetime, time as dt_time
from scalping import StockAnalyzer, PriceMonitor

def main():
    # API credentials
    API_KEY = os.getenv("ALPACA_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET")
    
    if not API_KEY or not SECRET_KEY:
        raise ValueError("API keys not set in environment variables")
    
    # Initialize analyzers
    analyzer = StockAnalyzer(API_KEY, SECRET_KEY)
    monitor = PriceMonitor(trading_client=analyzer.trading_client, 
                          data_client=analyzer.data_client)
    
    # File paths
    TICKER_FILE = "tickers.txt"
    PICKLE_FILE = "stocks.pickle"
    
    while True:
        current_time = datetime.now().time()
        
        # Only run during market hours (9:30 AM - 4:00 PM EST)
        if dt_time(9, 30) <= current_time <= dt_time(16, 00):
            try:
                print("\n=== Starting new analysis cycle ===")
                print(f"Time: {datetime.now()}")
                
                # Find best stock to monitor
                best_stock = analyzer.get_best_candidate(PICKLE_FILE, TICKER_FILE)
                
                if best_stock:
                    print(f"\nStarting price monitoring for {best_stock}")
                    
                    # Monitor for 5 minutes before re-analyzing
                    monitor_start = time.time()
                    monitor_duration = 300  # 5 minutes
                    
                    try:
                        monitor.monitor_symbol(best_stock)
                        while time.time() - monitor_start < monitor_duration:
                            # Check if market is still open
                            if datetime.now().time() > dt_time(16, 00):
                                print("Market closed, stopping monitoring")
                                break
                            time.sleep(1)
                    except Exception as e:
                        print(f"Error monitoring {best_stock}: {e}")
                else:
                    print("No suitable candidates found")
                    # Wait 1 minute before trying again
                    time.sleep(60)
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)
        else:
            # Outside market hours
            next_market_open = datetime.now().replace(
                hour=9, minute=30, second=0, microsecond=0)
            if current_time > dt_time(16, 00):
                next_market_open = next_market_open.replace(
                    day=next_market_open.day + 1)
            
            sleep_seconds = (next_market_open - datetime.now()).total_seconds()
            print(f"Market closed. Sleeping until {next_market_open}")
            time.sleep(sleep_seconds)

if __name__ == "__main__":
    main()