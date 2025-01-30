import os
import time
import concurrent.futures
from datetime import datetime
import pandas as pd
from scalping import ScalpingScreener
from pinger import PricePinger
from execution import OpportunityExecutor

def process_symbol(symbol, api_key, api_secret, max_depth=0.50):
    """Process a single symbol for opportunities"""
    try:
        pinger = PricePinger(api_key, api_secret)
        executor = OpportunityExecutor(api_key, api_secret)
        
        print(f"\nProcessing {symbol}...")
        
        # Find price levels with liquidity
        ping_results = pinger.ping_symbol(
            symbol,
            num_levels=10,
            max_depth=max_depth
        )
        
        if not ping_results.empty:
            # Look for filled ping orders
            filled_levels = ping_results[ping_results['filled']]
            
            results = []
            for _, row in filled_levels.iterrows():
                price_level = row['price_level']
                print(f"Found liquidity for {symbol} at {price_level}")
                
                # Execute opportunity at this price level
                result = executor.execute_opportunity(symbol, price_level)
                if result:
                    results.append(result)
            
            return results
        
        return []
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return []

def main():
    # Get API credentials
    API_KEY = os.getenv("ALPACA_KEY")
    API_SECRET = os.getenv("ALPACA_SECRET")
    
    if not API_KEY or not API_SECRET:
        print("Please set ALPACA_KEY and ALPACA_SECRET environment variables")
        exit(1)
    
    while True:
        try:
            print("\n" + "="*50)
            print(f"Starting new screening cycle at {datetime.now()}")
            print("="*50)
            
            # 1. Run screener to find top opportunities
            screener = ScalpingScreener(API_KEY, API_SECRET)
            opportunities = screener.screen_stocks(
                min_volume=100,  # Minimum average volume
                max_spread_pct=0.5  # Maximum spread percentage
            )
            
            if opportunities.empty:
                print("No opportunities found in screening")
                time.sleep(60)  # Wait 1 minute before next cycle
                continue
            
            # 2. Get top 5 symbols
            top_symbols = opportunities.head(5)['symbol'].tolist()
            print(f"\nTop opportunities found: {', '.join(top_symbols)}")
            
            # 3. Process symbols in parallel
            all_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Create futures for each symbol
                futures = {
                    executor.submit(
                        process_symbol, 
                        symbol, 
                        API_KEY, 
                        API_SECRET,
                        0.50  # max_depth
                    ): symbol for symbol in top_symbols
                }
                
                # Process completed futures
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        results = future.result()
                        if results:
                            all_results.extend(results)
                    except Exception as e:
                        print(f"Error in future for {symbol}: {e}")
            
            # 4. Summarize results
            if all_results:
                results_df = pd.DataFrame(all_results)
                print("\nTrading Results Summary:")
                summary = results_df.groupby('symbol').agg({
                    'profit_loss': ['sum', 'mean'],
                    'profit_loss_pct': ['mean']
                }).round(2)
                print(summary)
                
                # Save results to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_df.to_csv(f"trading_results_{timestamp}.csv", index=False)
            else:
                print("\nNo successful trades in this cycle")
            
            # 5. Wait before next cycle
            print("\nWaiting 5 seconds before next cycle...")
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying
            continue

if __name__ == "__main__":
    # Make sure environment variables are set
    import config
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")