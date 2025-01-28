# Standard library imports
import os
import pickle
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

# Third-party imports
import pandas as pd
import numpy as np
from alpaca.data import StockHistoricalDataClient
from alpaca.trading import TradingClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# Local imports
import config


# Basically just find what stocks we should look at
# Playing it safe right now with lower risk stocks, might have a smaller bid ask spread
#   Should add a sorter - maybe most volatile 


class StockAnalyzer:
    def __init__(self, api_key, secret_key):
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.min_spread = 0.001  # 0.1%
        self.max_spread = 0.01   # 1%
        self.max_volatility = 0.02  # 2%
    
    def create_pickle_from_txt(self, txt_file, pickle_file):
        """Create pickle file from text file containing tickers"""
        stock_dict = {}
        with open(txt_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        for ticker in tickers:
            stock_dict[ticker] = {
                'last_analysis': None,
                'signals': {},
                'historical_signals': []
            }
        with open(pickle_file, 'wb') as f:
            pickle.dump(stock_dict, f)
        return stock_dict
    
    def load_pickle(self, pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def save_pickle(self, data, pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
    
    def get_historical_data(self, symbol, lookback_minutes=60):
        """Get historical data and convert to proper DataFrame format"""
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(minutes=lookback_minutes)
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt
        )
        
        try:
            bars = self.data_client.get_stock_bars(request)
            
            # Convert tuple-style data to proper DataFrame
            data = []
            for bar in bars.data[symbol]:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            # Add volatility calculation
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None

    def get_current_spread(self, symbol):
        """Get current bid-ask spread"""
        try:
            quote = self.trading_client.get_latest_quote(symbol)
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            spread = (ask - bid) / bid
            return {
                'bid': bid,
                'ask': ask,
                'spread': spread
            }
        except Exception as e:
            print(f"Error getting spread for {symbol}: {e}")
            return None

    def check_signals(self, df, spread_data):
        """Enhanced signal checking with spread and volatility"""
        last_row = df.iloc[-1]
        signals = {
            'price_vs_vwap': 'ABOVE' if last_row['close'] > last_row['vwap'] else 'BELOW',
            'ema_cross': 'BULLISH' if last_row['ema9'] > last_row['ema20'] else 'BEARISH',
            'rsi': float(last_row['rsi']),
            'timestamp': last_row.name,
            'close': float(last_row['close']),
            'volume': float(last_row['volume']),
            'volatility': float(last_row['volatility']) if 'volatility' in df else None,
            'bid': spread_data['bid'] if spread_data else None,
            'ask': spread_data['ask'] if spread_data else None,
            'spread': spread_data['spread'] if spread_data else None,
            'signal': ""
        }
        
        # Original bullish conditions
        base_bullish = (
            signals['price_vs_vwap'] == 'ABOVE' and
            signals['ema_cross'] == 'BULLISH' and
            30 < signals['rsi'] < 70
        )
        
        # Add spread and volatility conditions
        spread_vol_good = (
            signals['spread'] is not None and
            signals['volatility'] is not None and
            self.min_spread <= signals['spread'] <= self.max_spread and
            signals['volatility'] <= self.max_volatility
        )
        
        bullish = base_bullish and spread_vol_good
        
        # Original bearish conditions (kept for completeness)
        bearish = (
            signals['price_vs_vwap'] == 'BELOW' and
            signals['ema_cross'] == 'BEARISH' and
            30 < signals['rsi'] < 70
        )
        
        signals['signal'] = 'BUY' if bullish else 'SELL' if bearish else 'NEUTRAL'
        return signals

    def analyze_stocks(self, pickle_file, txt_file=None):
        if txt_file and not self.load_pickle(pickle_file):
            stock_data = self.create_pickle_from_txt(txt_file, pickle_file)
        else:
            stock_data = self.load_pickle(pickle_file)
        
        current_time = datetime.now()
        good_candidates = []
        
        for ticker in stock_data.keys():
            try:
                print(f"\nAnalyzing {ticker}...")
                df = self.get_historical_data(ticker)
                
                if df is None or len(df) < 20:
                    print(f"Not enough data for {ticker}")
                    continue
                
                df = self.calculate_indicators(df)
                spread_data = self.get_current_spread(ticker)
                if spread_data is None:
                    continue
                    
                signals = self.check_signals(df, spread_data)
                
                # Store signals in pickle data
                stock_data[ticker]['last_analysis'] = current_time
                stock_data[ticker]['signals'] = signals
                stock_data[ticker]['historical_signals'].append({
                    'timestamp': current_time,
                    'signal': signals['signal'],
                    'price': signals['close']
                })
                
                if len(stock_data[ticker]['historical_signals']) > 100:
                    stock_data[ticker]['historical_signals'] = stock_data[ticker]['historical_signals'][-100:]
                
                # If this is a good candidate, add to our list
                if signals['signal'] == 'BUY' and spread_data['spread'] >= self.min_spread:
                    good_candidates.append({
                        'symbol': ticker,
                        'volatility': signals['volatility'],
                        'spread': spread_data['spread'],
                        'price': signals['close'],
                        'rsi': signals['rsi']
                    })
                    print(f"Found candidate: {ticker}")
                    print(f"Spread: {spread_data['spread']*100:.3f}%")
                    print(f"Volatility: {signals['volatility']*100:.2f}%")
                    print(f"RSI: {signals['rsi']:.2f}")
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                continue
        
        # Sort candidates by best combination of low volatility and good spread
        if good_candidates:
            good_candidates.sort(key=lambda x: (x['volatility'], -x['spread']))
        
        self.save_pickle(stock_data, pickle_file)
        return stock_data, good_candidates

    def get_best_candidate(self, pickle_file, txt_file=None):
        """Get the best stock to monitor based on all criteria"""
        _, candidates = self.analyze_stocks(pickle_file, txt_file)
        if candidates:
            best = candidates[0]
            print(f"\nBest candidate found: {best['symbol']}")
            print(f"Price: ${best['price']:.2f}")
            print(f"Spread: {best['spread']*100:.3f}%")
            print(f"Volatility: {best['volatility']*100:.2f}%")
            print(f"RSI: {best['rsi']:.2f}")
            return best['symbol']
        return None

# Class pinger
#   Want to ping in here
#       Can test ~ will pings actually work? am I going to need the dark pool?

#  process entry
#   want to buy/sell in here
#       sell almost immediately, might have to make sure liquidity is high 

# Maximize profits? how many trades do I have to do? risk (stop loss etc)

class PriceMonitor:
    def __init__(self, trading_client, data_client):
        self.trading_client = trading_client
        self.data_client = data_client
        self.last_account_check = 0
        self.cached_buying_power = 0
        self.api_calls = 0
        self.last_api_window = time.time()
        self.MAX_CALLS_PER_MIN = 200
        
    def track_api_call(self):
        """Track API calls and wait if necessary"""
        current_time = time.time()
        if current_time - self.last_api_window >= 60:
            self.api_calls = 0
            self.last_api_window = current_time
            
        self.api_calls += 1
        if self.api_calls >= self.MAX_CALLS_PER_MIN:
            sleep_time = 60 - (current_time - self.last_api_window)
            if sleep_time > 0:
                print(f"API rate limit reached, waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                self.api_calls = 0
                self.last_api_window = time.time()

    def get_cached_buying_power(self):
        """Get buying power with caching to reduce API calls"""
        current_time = time.time()
        if current_time - self.last_account_check >= 60:
            self.track_api_call()
            account = self.trading_client.get_account()
            self.cached_buying_power = float(account.buying_power)
            self.last_account_check = current_time
        return self.cached_buying_power

    def ping_price(self, symbol):
        """Ping market with price levels from 50% to 1% discount"""
        try:
            self.track_api_call()
            current_quote = self.trading_client.get_latest_quote(symbol)
            bid_price = float(current_quote.bid_price)
            ask_price = float(current_quote.ask_price)
            print(f"\nCurrent {symbol} bid: ${bid_price}, ask: ${ask_price}")
            
            # Create discount levels from 50% to 1%
            discounts = np.linspace(0.50, 0.01, 10)  # [0.50, 0.44, 0.39, 0.33, 0.28, 0.23, 0.17, 0.12, 0.06, 0.01]
            
            for discount in discounts:
                ping_price = round(bid_price * (1 - discount), 2)
                print(f"\nTrying {discount*100:.1f}% discount: ${ping_price}")
                
                self.track_api_call()
                ping_order = LimitOrderRequest(
                    symbol=symbol,
                    limit_price=ping_price,
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.IOC
                )
                
                response = self.trading_client.submit_order(ping_order)
                print(f"Order status: {response.status}")
                
                # If we get a fill, we found our price level
                if response.status in ['filled', 'partially_filled']:
                    print(f"Found fill at {discount*100:.1f}% discount!")
                    return ping_price
                    
            return None
            
        except Exception as e:
            print(f"Error pinging {symbol}: {e}")
            return None

    def execute_opportunity(self, symbol, ping_price):
        """Execute single large order at opportunity price"""
        try:
            buying_power = self.get_cached_buying_power()
            quantity = int(buying_power / ping_price)  # Use full buying power since we're only trading one stock
            
            if quantity <= 0:
                return None

            print(f"\nAttempting to buy {quantity} shares at ${ping_price}")
            self.track_api_call()
            buy_order = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                limit_price=ping_price,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.IOC
            )
            
            response = self.trading_client.submit_order(buy_order)
            
            if response.status in ['filled', 'partially_filled']:
                filled_qty = float(response.filled_qty)
                fill_price = float(response.filled_avg_price)
                print(f"Filled {filled_qty} shares at ${fill_price}")
                
                # Immediate market sell
                self.track_api_call()
                sell_order = MarketOrderRequest(
                    symbol=symbol,
                    qty=filled_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.IOC
                )
                
                sell_response = self.trading_client.submit_order(sell_order)
                if sell_response.status == 'filled':
                    sell_price = float(sell_response.filled_avg_price)
                    profit = (sell_price - fill_price) * filled_qty
                    print(f"Sold {filled_qty} shares at ${sell_price}")
                    print(f"Profit: ${profit:.2f}")
                
                return {
                    'buy_order': response,
                    'sell_order': sell_response,
                    'filled_qty': filled_qty,
                    'profit': profit
                }
            
            return None
            
        except Exception as e:
            print(f"Error executing orders: {e}")
            return None

    def monitor_symbol(self, symbol):
        """Monitor single symbol continuously"""
        print(f"\nStarting monitoring of {symbol}")
        
        while True:
            try:
                # Find opportunity price through pinging
                opportunity_price = self.ping_price(symbol)
                
                if opportunity_price:
                    print(f"\nOpportunity found for {symbol} at ${opportunity_price}")
                    trade_result = self.execute_opportunity(symbol, opportunity_price)
                    if trade_result:
                        print(f"\nTrade completed for {symbol}")
                        print(f"Profit: ${trade_result['profit']:.2f}")
                
                # Add delay between attempts to stay under rate limit
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)

def main():
    import os
    API_KEY = os.getenv("ALPACA_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET")
    
    # Example usage
    screener = StockScreener(API_KEY, SECRET_KEY)
    
    # Sample list of stocks (replace with your list)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    print("Screening stocks...")
    results = screener.screen_stocks(symbols)
    
    print("\nTop candidates (sorted by lowest volatility):")
    for r in results:
        print(f"\n{r['symbol']}:")
        print(f"Volatility: {r['volatility']*100:.2f}%")
        print(f"Spread: {r['spread']*100:.3f}%")
        print(f"Price: ${r['avg_price']:.2f}")

if __name__ == "__main__":
    main()
        



        ## When I run out of things to say - maybe we can go get some coffee and talk about our life ambitions and if it gets stale we do work - ill drive