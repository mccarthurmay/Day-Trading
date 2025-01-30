from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import time
from datetime import datetime
import pandas as pd

class OpportunityExecutor:
    def __init__(self, api_key, api_secret):
        self.trading_client = TradingClient(api_key, api_secret)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        
    def get_buying_power(self):
        """Get current available buying power"""
        try:
            account = self.trading_client.get_account()
            return float(account.buying_power)
        except Exception as e:
            print(f"Error getting buying power: {e}")
            return 0.0
            
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
            
    def calculate_max_shares(self, price):
        """Calculate maximum shares we can buy at given price"""
        buying_power = self.get_buying_power()
        max_shares = int(buying_power / price)
        return max_shares
        
    def place_large_buy(self, symbol, price):
        """Place a large limit buy order at the specified price"""
        max_shares = self.calculate_max_shares(price)
        
        if max_shares < 1:
            print("Insufficient buying power for trade")
            return None
            
        try:
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=max_shares,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.IOC,  # Immediate-or-Cancel
                limit_price=price
            )
            
            order = self.trading_client.submit_order(order_data)
            
            # Wait briefly for fill
            time.sleep(0.1)
            
            # Get final order status
            order = self.trading_client.get_order_by_id(order.id)
            
            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
            
            return {
                'order_id': order.id,
                'filled_qty': filled_qty,
                'filled_price': filled_price
            }
            
        except Exception as e:
            print(f"Error placing buy order: {e}")
            return None
            
    def place_market_sell(self, symbol, qty):
        """Place market sell order for specified quantity"""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            
            # Wait briefly for fill
            time.sleep(0.1)
            
            # Get final order status
            order = self.trading_client.get_order_by_id(order.id)
            
            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
            
            return {
                'order_id': order.id,
                'filled_qty': filled_qty,
                'filled_price': filled_price
            }
            
        except Exception as e:
            print(f"Error placing sell order: {e}")
            return None
    
    def execute_opportunity(self, symbol, price_level):
        """Execute full trading opportunity
        
        Args:
            symbol (str): Stock symbol
            price_level (float): Price level where ping order filled
            
        Returns:
            dict: Trade results including P&L
        """
        print(f"\nExecuting opportunity for {symbol} at {price_level}")
        
        # Step 1: Place large buy order
        buy_order = self.place_large_buy(symbol, price_level)
        if not buy_order or buy_order['filled_qty'] == 0:
            print("Buy order did not fill")
            return None
            
        print(f"Bought {buy_order['filled_qty']} shares at {buy_order['filled_price']}")
        
        # Step 2: Place market sell order
        sell_order = self.place_market_sell(symbol, buy_order['filled_qty'])
        if not sell_order:
            print("Error placing sell order")
            return None
            
        print(f"Sold {sell_order['filled_qty']} shares at {sell_order['filled_price']}")
        
        # Calculate P&L
        buy_cost = buy_order['filled_qty'] * buy_order['filled_price']
        sell_proceeds = sell_order['filled_qty'] * sell_order['filled_price']
        profit_loss = sell_proceeds - buy_cost
        profit_loss_pct = (profit_loss / buy_cost) * 100
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'buy_qty': buy_order['filled_qty'],
            'buy_price': buy_order['filled_price'],
            'sell_qty': sell_order['filled_qty'],
            'sell_price': sell_order['filled_price'],
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        }
        
        print(f"\nTrade Results for {symbol}:")
        print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
        
        return results

def execute_ping_opportunities(api_key, api_secret, symbols, num_levels=10, max_depth=0.50):
    """Main function to find and execute opportunities
    
    Args:
        api_key (str): Alpaca API key
        api_secret (str): Alpaca API secret
        symbols (list): List of symbols to check
        num_levels (int): Number of price levels to test
        max_depth (float): Maximum price drop percentage
    """
    from pinger import PricePinger
    
    pinger = PricePinger(api_key, api_secret)
    executor = OpportunityExecutor(api_key, api_secret)
    
    trade_results = []
    
    for symbol in symbols:
        print(f"\nChecking {symbol} for opportunities...")
        
        # Find price levels with liquidity
        ping_results = pinger.ping_symbol(symbol, num_levels, max_depth)
        
        if not ping_results.empty:
            # Look for filled ping orders
            filled_levels = ping_results[ping_results['filled']]
            
            for _, row in filled_levels.iterrows():
                price_level = row['price_level']
                print(f"Found liquidity at {price_level}")
                
                # Execute opportunity at this price level
                result = executor.execute_opportunity(symbol, price_level)
                if result:
                    trade_results.append(result)
                    
    # Convert results to DataFrame
    if trade_results:
        results_df = pd.DataFrame(trade_results)
        print("\nAll Trade Results:")
        print(results_df[['symbol', 'profit_loss', 'profit_loss_pct']])
        return results_df
    else:
        print("No trade opportunities found")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    import os
    
    API_KEY = os.getenv("ALPACA_KEY")
    API_SECRET = os.getenv("ALPACA_SECRET")
    
    if not API_KEY or not API_SECRET:
        print("Please set ALPACA_KEY and ALPACA_SECRET environment variables")
        exit(1)
        
    # Test with a few symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    results = execute_ping_opportunities(
        API_KEY,
        API_SECRET,
        symbols,
        num_levels=10,
        max_depth=0.50
    )