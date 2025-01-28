# Standard library imports
import os
import pickle
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np
from alpaca.data import StockHistoricalDataClient
from alpaca.trading import TradingClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Local imports
import config

class StockAnalyzer:
    def __init__(self, api_key, secret_key):
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
    
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
        return df
    
    def calculate_indicators(self, df):
        """Calculate VWAP, EMAs, and RSI"""
        try:
            # Calculate VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Calculate EMAs
            df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            print("DataFrame head:", df.head())
            print("DataFrame columns:", df.columns)
            raise e
    
    def check_signals(self, df):
        """Check for trading signals based on indicators"""
        last_row = df.iloc[-1]
        signals = {
            'price_vs_vwap': 'ABOVE' if last_row['close'] > last_row['vwap'] else 'BELOW',
            'ema_cross': 'BULLISH' if last_row['ema9'] > last_row['ema20'] else 'BEARISH',
            'rsi': float(last_row['rsi']),
            'timestamp': last_row.name,
            'close': float(last_row['close']),
            'volume': float(last_row['volume']),
            'signal': ""
        }
        
        bullish = (
            signals['price_vs_vwap'] == 'ABOVE' and
            signals['ema_cross'] == 'BULLISH' and
            30 < signals['rsi'] < 70
        )
        
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
        
        for ticker in stock_data.keys():
            try:
                print(f"Analyzing {ticker}...")  # Debug print
                df = self.get_historical_data(ticker)
                
                if len(df) < 20:
                    print(f"Not enough data for {ticker}")
                    continue
                
                df = self.calculate_indicators(df)
                signals = self.check_signals(df)
                
                stock_data[ticker]['last_analysis'] = current_time
                stock_data[ticker]['signals'] = signals
                stock_data[ticker]['historical_signals'].append({
                    'timestamp': current_time,
                    'signal': signals['signal'],
                    'price': signals['close']
                })
                
                if len(stock_data[ticker]['historical_signals']) > 100:
                    stock_data[ticker]['historical_signals'] = stock_data[ticker]['historical_signals'][-100:]
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                continue
        
        self.save_pickle(stock_data, pickle_file)
        return stock_data

# Debug function to print DataFrame structure
def print_df_info(df):
    print("\nDataFrame Info:")
    print("Columns:", df.columns)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)

if __name__ == "__main__":
    import os
    API_KEY = os.getenv("ALPACA_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET")
    TICKER_FILE = "tickers.txt"
    PICKLE_FILE = "stocks.pickle"
    
    if not API_KEY or not SECRET_KEY:
        raise ValueError("ALPACA_KEY and ALPACA_SECRET environment variables must be set")
    
    analyzer = StockAnalyzer(API_KEY, SECRET_KEY)
    results = analyzer.analyze_stocks(PICKLE_FILE, TICKER_FILE)
    
    brokenTickers = []

    for ticker, data in results.items():
        current_signal = data['signals']
        try:
            if current_signal['signal'] != 'NEUTRAL':
                print(f"\n{ticker} Analysis:")
                print(f"Signal: {current_signal['signal']}")
                print(f"Price vs VWAP: {current_signal['price_vs_vwap']}")
                print(f"EMA Cross: {current_signal['ema_cross']}")
                print(f"RSI: {current_signal['rsi']:.2f}")
                print(f"Price: ${current_signal['close']:.2f}")
                print(f"Time: {current_signal['timestamp']}") 
        except:
            brokenTickers.append(ticker)
    if len(brokenTickers) > 0:
        print(f"These tickers are broken: {brokenTickers}")
        



        ## When I run out of things to say - maybe we can go get some coffee and talk about our life ambitions and if it gets stale we do work - ill drive