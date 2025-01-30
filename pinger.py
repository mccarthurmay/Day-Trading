from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import numpy as np
import time
from datetime import datetime
import pandas as pd
import os
import config

class PricePinger:
    def __init__(self, api_key, api_secret):
        self.trading_client = TradingClient(api_key, api_secret)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        
    def get_current_quote(self, symbol):
        """Get current bid/ask for a symbol"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'bid_size': float(quote.bid_size),
                    'ask_size': float(quote.ask_size)
                }
            return None
            
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None
            
    def generate_ping_levels(self, current_bid, num_levels=10, max_depth=0.50):
        """Generate price levels for pinging
        
        Args:
            current_bid (float): Current bid price
            num_levels (int): Number of price levels to test
            max_depth (float): Maximum price drop as percentage (0.50 = 50% below bid)
            
        Returns:
            list: Price levels to test
        """
        # Create array of percentages from max_depth to 0.01
        percentages = np.linspace(max_depth, 0.01, num_levels)
        
        # Calculate prices at each percentage below bid
        prices = [current_bid * (1 - pct) for pct in percentages]
        
        # Round to appropriate number of decimal places based on price
        rounded_prices = []
        for price in prices:
            if price < 1:
                rounded = round(price, 4)
            elif price < 10:
                rounded = round(price, 3)
            elif price < 100:
                rounded = round(price, 2)
            else:
                rounded = round(price, 2)
            rounded_prices.append(rounded)
            
        return rounded_prices
        
    def ping_price_level(self, symbol, price, shares=1):
        """Test a single price level with an IOC limit order
        
        Args:
            symbol (str): Stock symbol
            price (float): Limit price to test
            shares (int): Number of shares for test order
            
        Returns:
            dict: Order result including fill status and price
        """
        # Create IOC limit order
        order_data = LimitOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.IOC,
            limit_price=price
        )
        
        try:
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            # Wait briefly for fill/cancel
            time.sleep(0.1)
            
            # Get updated order status
            order = self.trading_client.get_order_by_id(order.id)
            
            # Convert quantities to float for numeric comparison
            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
            
            return {
                'price_level': price,
                'filled': filled_qty > 0,
                'filled_qty': filled_qty,
                'filled_price': filled_price
            }
            
        except Exception as e:
            print(f"Error pinging price {price} for {symbol}: {e}")
            return None
            
    def ping_symbol(self, symbol, num_levels=10, max_depth=0.50):
        """Test multiple price levels for a symbol
        
        Args:
            symbol (str): Stock symbol to test
            num_levels (int): Number of price levels to test
            max_depth (float): Maximum price drop as percentage
            
        Returns:
            pd.DataFrame: Results of price level tests
        """
        # Get current quote
        quote = self.get_current_quote(symbol)
        if not quote:
            return pd.DataFrame()  # Return empty DataFrame instead of None
            
        current_bid = quote['bid']
        
        # Generate price levels
        price_levels = self.generate_ping_levels(
            current_bid,
            num_levels=num_levels,
            max_depth=max_depth
        )
        
        results = []
        
        # Test each price level
        for price in price_levels:
            result = self.ping_price_level(symbol, price)
            if result:
                result['timestamp'] = datetime.now()
                results.append(result)
                
            # Brief pause between orders
            time.sleep(0.5)
            
        # Convert results to DataFrame
        if results:
            df = pd.DataFrame(results)
            df['pct_below_bid'] = ((current_bid - df['price_level']) / current_bid) * 100
            return df
        
        return pd.DataFrame()  # Return empty DataFrame if no results

# Example usage:
if __name__ == "__main__":
    import os
    
    # Get API credentials from environment variables
    API_KEY = os.getenv("ALPACA_KEY")
    API_SECRET = os.getenv("ALPACA_SECRET")
    
    if not API_KEY or not API_SECRET:
        print("Please set ALPACA_KEY and ALPACA_SECRET environment variables")
        exit(1)
    
    # Create pinger
    pinger = PricePinger(API_KEY, API_SECRET)
    
    # Test price levels for a symbol
    symbol = "AAPL"
    results = pinger.ping_symbol(
        symbol,
        num_levels=10,
        max_depth=0.50
    )
    
    if not results.empty:
        print(f"\nHidden Liquidity Test Results for {symbol}:")
        print(results[['price_level', 'pct_below_bid', 'filled', 'filled_qty', 'filled_price']])
    else:
        print(f"No results found for {symbol}")