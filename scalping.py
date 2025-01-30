import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pytz
import os
import config

class ScalpingScreener:
    def __init__(self, api_key, api_secret, batch_size=100):
        self.trading_client = TradingClient(api_key, api_secret)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        self.batch_size = batch_size
        
    def get_tickers_from_file(self, filename='tickers.txt'):
        """Read tickers from text file"""
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f.readlines()]
        return tickers

    def get_historical_data_batch(self, symbols):
        """Get historical bars for analysis"""
        # Get data from 1 hour ago to account for 15-min delay
        end_dt = datetime.now(pytz.UTC) - timedelta(minutes=15)
        start_dt = end_dt - timedelta(hours=5)
        
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt
        )
        
        try:
            bars = self.data_client.get_stock_bars(bars_request)
            return bars
        except Exception as e:
            print(f"Error fetching historical data for batch: {e}")
            return None

    def analyze_historical_bars(self, df):
        """Analyze bar data for patterns and metrics"""
        if len(df) < 3:  # Need at least 3 bars for minimal analysis
            return None
        
        # Calculate various metrics
        df['volume'] = df.apply(lambda row: next((value for key, value in row if key == 'volume'), None), axis=1)
        df['close'] = df.apply(lambda row: next((value for key, value in row if key == 'close'), None), axis=1)
        df['high'] = df.apply(lambda row: next((value for key, value in row if key == 'high'), None), axis=1)
        df['low'] = df.apply(lambda row: next((value for key, value in row if key == 'low'), None), axis=1)

        vol_mean = df['volume'].mean()
        vol_std = df['volume'].std()
        price_mean = df['close'].mean()
        price_std = df['close'].std()
        
        # Calculate average true range (ATR)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        atr = df['tr'].mean()
        
        # Calculate price momentum
        momentum = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        
        # Calculate volume momentum
        vol_momentum = (df['volume'].iloc[-1] - df['volume'].mean()) / df['volume'].std()
        
        return {
            'last_price': df['close'].iloc[-1],
            'price_mean': price_mean,
            'price_std': price_std,
            'volume_mean': vol_mean,
            'volume_std': vol_std,
            'atr': atr,
            'momentum': momentum,
            'volume_momentum': vol_momentum,
            'avg_spread': (df['high'] - df['low']).mean(),
            'avg_spread_pct': ((df['high'] - df['low']) / df['low']).mean() * 100
        }

    def screen_stocks(self, min_volume=100000, max_spread_pct=0.1):
        """Screen stocks based on historical patterns"""
        symbols = self.get_tickers_from_file()
        #print(f"Screening {len(symbols)} stocks from tickers.txt...")
        
        results = []
        
        # Process symbols in batches
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i + self.batch_size]
            #print(f"Processing batch {i//self.batch_size + 1} of {len(symbols)//self.batch_size + 1}")
            
            bars = self.get_historical_data_batch(batch)
            if not bars:
                continue
            # Process each symbol in the batch
            for symbol in batch:
                try:
                    #print(f"Processing {symbol}")
                    symbol_data = bars[symbol]
                    if symbol_data:
                        # Get the DataFrame directly from the bars object
                        df = pd.DataFrame(symbol_data)
                    #print(f"Processing {symbol} with {len(df)} bars")
                    analysis = self.analyze_historical_bars(df)
                    #print(analysis)
                    if analysis and analysis['volume_mean'] >= min_volume and analysis['avg_spread_pct'] <= max_spread_pct:
                        
                        # Calculate opportunity score
                        score = self.calculate_opportunity_score(
                            spread_pct=analysis['avg_spread_pct'],
                            volume=analysis['volume_mean'],
                            momentum=analysis['momentum'],
                            volatility=analysis['price_std'],
                            volume_momentum=analysis['volume_momentum']
                        )
                        #print(score)
                        
                        results.append({
                            'symbol': symbol,
                            'score': score,
                            'price': analysis['last_price'],
                            'volume': analysis['volume_mean'],
                            'spread_pct': analysis['avg_spread_pct'],
                            'momentum': analysis['momentum'],
                            'volume_momentum': analysis['volume_momentum'],
                            'atr': analysis['atr']
                        })
                except KeyError as e:
                    continue
    
        # Convert to DataFrame and sort by score
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('score', ascending=False)
        
        return results_df
    
    def calculate_opportunity_score(self, spread_pct, volume, momentum, volatility, volume_momentum):
        """Calculate composite score for scalping opportunity"""
        # Normalize metrics to 0-1 scale
        spread_score = 1 - (spread_pct / 0.1)  # Lower spread is better
        volume_score = min(volume / 100000, 1.0)  # Higher volume is better
        momentum_score = abs(momentum)  # Strong momentum either direction
        volatility_score = min(volatility, 1.0)  # Some volatility is good
        volume_momentum_score = max(min(volume_momentum, 1.0), 0)  # Rising volume is good
        
        # Weighted average of components
        weights = {
            'spread': 0.3,
            'volume': 0.2,
            'momentum': 0.2,
            'volatility': 0.15,
            'volume_momentum': 0.15
        }
        
        score = (
            weights['spread'] * spread_score +
            weights['volume'] * volume_score +
            weights['momentum'] * momentum_score +
            weights['volatility'] * volatility_score +
            weights['volume_momentum'] * volume_momentum_score
        )
        
        return score

# Example usage
if __name__ == "__main__":
    API_KEY = os.getenv("ALPACA_KEY")
    API_SECRET = os.getenv("ALPACA_SECRET")
    
    screener = ScalpingScreener(API_KEY, API_SECRET)
    
    # Run screening
    results = screener.screen_stocks(min_volume=50, max_spread_pct=10)
    
    # Display top 10 opportunities
    print("\nTop 10 Scalping Opportunities:")
    if not results.empty:
        print(results.head(10)[['symbol', 'score', 'price', 'spread_pct', 'volume', 'momentum']])
    else:
        print("No opportunities found matching criteria")