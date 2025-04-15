# -*- coding: utf-8 -*-
"""
Enhanced Kraken Momentum Trading Bot with Order Placement
"""

import time
import numpy as np
from datetime import datetime, timedelta
import requests
import threading
from collections import deque, defaultdict
import ccxt
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")


# Initialize Kraken exchange
exchange = ccxt.kraken({
    'apiKey': os.getenv('KRAKEN_API_KEY'),
    'secret': os.getenv('KRAKEN_API_SECRET'),
    'enableRateLimit': True,
    'rateLimit': 3000,  # ms between requests
    'options': {
        'adjustForTimeDifference': True,
        'createMarketBuyOrderRequiresPrice': False
    }
})

# Configuration
INITIAL_BALANCE = 333  # USD
RISK_PER_TRADE = 0.02  # 2% of portfolio
STOP_LOSS = -20  # Percentage
TAKE_PROFIT = 2  # Percentage
TRAILING_TAKE_PROFIT = 0.02  # 2% trailing
MAX_POSITIONS = 3
DEFAULT_MOMENTUM_PERIOD = 5
CHECK_INTERVAL = 1800  # 15 minutes
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
ATH_THRESHOLD = 0.80  # Max 5% below ATH to consider
VOLATILITY_WINDOW = 10  # Period for calculating volatility
MIN_MOMENTUM_PERIOD = 3
MAX_MOMENTUM_PERIOD = 10
TIMEFRAMES = ['1h', '4h', '1d']  # Multi-Timeframe Analysis
BREAK_EVEN_THRESHOLD = 1.5  # Move stop-loss to break-even after 1.5% gain
STOP_LOSS_MULTIPLIER = 1.5  # ATR-based dynamic stop loss

class KrakenMomentumTrader:
    def __init__(self):
        self.portfolio = {
            'USD': INITIAL_BALANCE,
            'positions': {},
            'last_rebalance': datetime.now()
        }
        self.momentum_period = DEFAULT_MOMENTUM_PERIOD
        self.weights = np.linspace(1, 2, self.momentum_period)
        self.cache = defaultdict(dict)
        self.lock = threading.Lock()
        self.request_times = deque(maxlen=15)  # Kraken's public call limit
        
        # Load markets and filter USD pairs
        self.load_markets()
        
    def load_markets(self):
        """Load and cache available markets"""
        try:
            markets = exchange.load_markets()
            self.usd_pairs = [markets[symbol]['id'] for symbol in markets 
                            if markets[symbol]['quote'] == 'USD' 
                            and markets[symbol]['active']]
            for coin in ['ALICEUSD',
                         'REZUSD',
                         'EURRUSD',
                         'USDQUSD',
                         'OMNIUSD',
                         'EURQUSD',
                         'ANLOGUSD',
                         'WALUSD',
                         'LSETHUSD',
                         'USTUSD',
                         'C98USD',
                         'USDRUSD',
                         'PAXGUSD',
                         'DAIUSD',
                         'LAYERUSD',
                         'USDTZUSD',
                         'EUROPUSD',
                         'USDGUSD',
                         'USDCUSD',
                         'ZGBPZUSD',
                         'ZEURZUSD',
                         'PYUSDUSD',
                         'USDDUSD',
                         'RLUSDUSD',
                         'EURTUSD',
                         'TUSDUSD',
                         'L3USD',
                         'AUDUSD',
                         'KUSD',
                         'HDXUSD',
                         'REQUSD',
                         'DUCKUSD',
                         'TBTCUSD',
                         'WBTCUSD',
                         'PROMPTUSD',
                         'FHEUSD',
                         'AVAAIUSD']:
                if coin in self.usd_pairs:
                    self.usd_pairs.remove(coin)
            self.usd_pairs = self.usd_pairs
            print(f"Loaded {len(self.usd_pairs)} trading pairs")
        except Exception as e:
            print(f"Error loading markets: {str(e)}")
            self.usd_pairs = []

    def rate_limited_request(self, func, *args, **kwargs):
        """Handle rate limiting with exponential backoff"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Enforce rate limit
                now = time.time()
                if len(self.request_times) >= 15 and (now - self.request_times[0]) < 10:
                    sleep_time = 10 - (now - self.request_times[0])
                    time.sleep(sleep_time)
                
                result = func(*args, **kwargs)
                self.request_times.append(time.time())
                return result
                
            except ccxt.RateLimitExceeded:
                wait = min(10, (attempt + 1) * 2)
                print(f"Rate limit exceeded, waiting {wait} seconds...")
                time.sleep(wait)
            except ccxt.NetworkError as e:
                print(f"Network error: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)
            except ccxt.ExchangeError as e:
                print(f"Exchange error: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
            except Exception as e:
                print(f"API error: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
        
        raise Exception(f"Request failed after {max_retries} attempts")

    def place_order(self, pair, side, amount, price=None, order_type='limit'):
        """
        Place an order on Kraken
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            side: 'buy' or 'sell'
            amount: Quantity to trade
            price: Price for limit orders (None for market)
            order_type: 'limit' or 'market'
        Returns:
            order_info: Dictionary with order details or None if failed
        """
        try:
            params = {}
            
            if order_type == 'limit' and price is not None:
                params['price'] = self.safe_float(price)
                params['ordertype'] = 'limit'
            else:
                params['ordertype'] = 'market'
            
            # Convert amount to string to avoid precision issues
            amount_str = format(self.safe_float(amount), '.8f').rstrip('0').rstrip('.')
            
            print(f"Placing {order_type} {side} order for {amount_str} {pair} at {price}")
            
            order = exchange.create_order(
                pair,
                order_type,
                side,
                amount_str,
                price,
                params)
            print(order)
            if order:
                order_info = {
                    'id': order.get('id'),
                    'pair': pair,
                    'side': side,
                    'type': order.get('type', 'limit'),
                    'amount': self.safe_float(order.get('amount',0)),
                    'price': self.safe_float(order.get('price',0)),
                    'filled': self.safe_float(order.get('filled',0)),
                    'status': order.get('status', 'Done'),
                    'timestamp': datetime.now(),
                    'fee': (order.get('fee') or {}).get('cost', 0)
                }
                #self.send_alert(f"Order executed: {order_info}")
                print(f"Order executed: {order_info}")
                return order_info
            
        except ccxt.InsufficientFunds as e:
            self.send_alert(f"⚠️ Insufficient funds for {side} order: {pair} {amount}@{price}")
            print(f"Insufficient funds: {str(e)}")
        except ccxt.InvalidOrder as e:
            self.send_alert(f"⚠️ Invalid order: {pair} {side} {amount}@{price}")
            print(f"Invalid order: {str(e)}")
        except Exception as e:
            self.send_alert(f"⚠️ Order failed: {pair} {side} {amount}@{price}, Order error: {str(e)}")
            print(f"Order error: {str(e)}")
        
        return None

    def send_alert(self, message):
        """Send Telegram notification"""
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        try:
            requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': message}, timeout=5)
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")

    def safe_float(self, value):
        """Safely convert any value to float"""
        try:
            if isinstance(value, (list, tuple)):
                return float(value[0]) if len(value) > 0 else 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def dynamic_momentum_period(self, pair):
        try:
            ohlcv = self.rate_limited_request(exchange.fetch_ohlcv, pair, timeframe='1h', limit=VOLATILITY_WINDOW)
            prices = [candle[4] for candle in ohlcv]
            volatility = np.std(np.diff(prices))
            period = int(MAX_MOMENTUM_PERIOD - (volatility * (MAX_MOMENTUM_PERIOD - MIN_MOMENTUM_PERIOD)))
            period = max(MIN_MOMENTUM_PERIOD, min(period, MAX_MOMENTUM_PERIOD))
            return period
        except Exception as e:
            print(f"Volatility calculation error: {str(e)}")
            return DEFAULT_MOMENTUM_PERIOD

    def calculate_atr(self, pair, period=14):
        """Calculate the Average True Range (ATR) for dynamic stop-loss."""
        try:
            ohlcv = self.rate_limited_request(exchange.fetch_ohlcv, pair, timeframe='1d', limit=period+1)
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])
            closes = np.array([x[4] for x in ohlcv])
            tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
            atr = np.mean(tr)
            return atr
        except Exception as e:
            print(f"ATR calculation error for {pair}: {str(e)}")
            return None


    def calculate_momentum(self, pair):
         """Calculate weighted momentum score with caching, dynamic momentum period, and Multi-Timeframe Analysis"""
         scores = []
         
         for timeframe in TIMEFRAMES:
             cache_key = f"ohlcv_{pair}_{timeframe}"
             cached = self.cache.get(cache_key)
     
             # Determine momentum period dynamically and adjust weights
             self.momentum_period = self.dynamic_momentum_period(pair)
             self.weights = np.linspace(1, 2, self.momentum_period - 1)  # Adjust weights length
     
             if cached and time.time() - cached['timestamp'] < 1800:  # 30 min cache
                 ohlcv = cached['data']
             else:
                 try:
                     ohlcv = self.rate_limited_request(
                         exchange.fetch_ohlcv,
                         pair,
                         timeframe=timeframe,
                         limit=self.momentum_period
                     )
                     self.cache[cache_key] = {
                         'data': ohlcv,
                         'timestamp': time.time()
                     }
                 except Exception as e:
                     print(f"Error fetching OHLCV for {pair} on {timeframe}: {str(e)}")
                     continue
     
             if len(ohlcv) < self.momentum_period:
                 continue
     
             # Extract and convert data safely
             closes = np.array([self.safe_float(x[4]) for x in ohlcv])
             volumes = np.array([self.safe_float(x[5]) for x in ohlcv])
     
             # Calculate momentum components
             returns = np.diff(closes) / closes[:-1]
     
             # Ensure consistent length for weights and returns
             if len(returns) != len(self.weights):
                 print(f"Warning: Inconsistent lengths for weights and returns on {pair} in {timeframe}")
                 continue
     
             weighted_returns = returns * self.weights
             price_momentum = np.sum(weighted_returns) * 100
     
             # Volume adjustment
             volume_change = np.log(volumes[-1] / np.mean(volumes[:-3]))
     
             scores.append(price_momentum + volume_change * 20)
     
         return np.mean(scores) if scores else None


    def is_near_ath(self, pair):
        try:
            ohlcv = exchange.fetch_ohlcv(pair, timeframe='1d', limit=365)
            closes = np.array([x[4] for x in ohlcv])
            current_price = closes[-1]
            ath_price = np.max(closes)
            return current_price >= ath_price * (1 - ATH_THRESHOLD)
        except Exception as e:
            print(f"Error checking ATH for {pair}: {str(e)}")
            return False

    def get_top_pairs(self):
        BATCH_SIZE = 10
        scores = []

        for i in range(0, len(self.usd_pairs), BATCH_SIZE):
            batch = self.usd_pairs[i:i + BATCH_SIZE]
            batch_scores = []

            for pair in batch:
                try:
                    if self.is_near_ath(pair):
                        momentum = self.calculate_momentum(pair)
                        if momentum is not None:
                            batch_scores.append((pair, momentum))
                except Exception as e:
                    print(f"Error processing {pair}: {str(e)}")

            scores.extend(batch_scores)
            time.sleep(1)

        return sorted(scores, key=lambda x: x[1], reverse=True)[:MAX_POSITIONS]

    def calculate_position_size(self, price):
        """Risk-based position sizing"""
        with self.lock:
            total_value = self.portfolio['USD']
            for pos in self.portfolio['positions'].values():
                ticker = self.rate_limited_request(exchange.fetch_ticker, pos['pair'])
                current_price = self.safe_float(ticker['last'])
                total_value += pos['quantity'] * current_price
            
            risk_amount = total_value * RISK_PER_TRADE
            return risk_amount / price if price > 0 else 0

    def execute_trade(self, pair, amount, side):
        """Execute trade with proper order placement"""
        try:
            # Get current market price
            ticker = self.rate_limited_request(exchange.fetch_ticker, pair)
            price = self.safe_float(ticker['ask'] if side == 'buy' else ticker['bid'])
            
            if price <= 0 or amount <= 0:
                return None
                
            # Place the order
            order = self.place_order(
                pair=pair,
                side=side,
                amount=amount,
                price=price,
                order_type='limit'
            )
            if order:
                return {
                    'pair': pair,
                    'side': side,
                    'price': order['price'],
                    'quantity': order['amount'],
                    'timestamp': order['timestamp'],
                    'stop_loss': order['price'] * (1 + STOP_LOSS/100),
                    'take_profit': order['price'] * (1 + TAKE_PROFIT/100),
                    'highest_price': order['price'],
                    'order_id': order['id']
                }
            return None
            
        except Exception as e:
            self.send_alert(f"Trade execution failed for {pair}: {str(e)}")
            return None

    def check_order_status(self, order_id, pair):
        """Check status of an existing order"""
        try:
            order = self.rate_limited_request(exchange.fetch_order, order_id, pair)
            return {
                'status': order.get('status'),
                'filled': self.safe_float(order.get('filled')),
                'remaining': self.safe_float(order.get('remaining'))
            }
        except Exception as e:
            print(f"Error checking order status: {str(e)}")
            return None

    def manage_risk(self, position):
        """Dynamic risk management with trailing stop, break-even logic, and ATR-based stop loss."""
        try:
            ticker = self.rate_limited_request(exchange.fetch_ticker, position['pair'])
            current_price = self.safe_float(ticker['last'])

            # Update highest price
            position['highest_price'] = max(position['highest_price'], current_price)

            # ATR-based Stop Loss Calculation
            atr = self.calculate_atr(position['pair'])
            if atr:
                dynamic_stop_loss = position['price'] - (STOP_LOSS_MULTIPLIER * atr)
                position['stop_loss'] = max(position['stop_loss'], dynamic_stop_loss)

            # Break-Even Logic: Move stop-loss to entry price after a threshold gain
            if current_price >= position['price'] * (1 + BREAK_EVEN_THRESHOLD / 100):
                position['stop_loss'] = position['price']

            # Trailing Stop Logic
            new_stop = position['highest_price'] * (1 - TRAILING_TAKE_PROFIT)
            position['stop_loss'] = max(position['stop_loss'], new_stop)

            # Check exit conditions
            if current_price <= position['stop_loss']:
                return 'stop_loss'
            if current_price >= position['take_profit']:
                return 'take_profit'
            return None
        except Exception as e:
            print(f"Risk management error: {str(e)}")
            return None

    def rebalance_portfolio(self):
        """Portfolio rebalancing with proper locking"""
        # Close positions
        to_remove = []
        for pair, position in list(self.portfolio['positions'].items()):
            exit_reason = self.manage_risk(position)
            if exit_reason:
                sell_order = self.execute_trade(pair, position['quantity'], 'sell')
                if sell_order:
                    with self.lock:
                        self.portfolio['USD'] += sell_order['quantity'] * sell_order['price']
                        to_remove.append(pair)
                    
                    profit_pct = (sell_order['price'] - position['price'])/position['price']*100
                    msg = (f"Closed {pair} ({exit_reason})\n"
                          f"Profit: {profit_pct:.2f}%\n"
                          f"Duration: {(datetime.now() - position['timestamp']).days}d")
                    self.send_alert(msg)
        
        with self.lock:
            for pair in to_remove:
                self.portfolio['positions'].pop(pair, None)
        
        # Open new positions
        if len(self.portfolio['positions']) < MAX_POSITIONS:
            top_pairs = self.get_top_pairs()
            for pair, score in top_pairs:
                if pair not in self.portfolio['positions']:
                    ticker = self.rate_limited_request(exchange.fetch_ticker, pair)
                    price = self.safe_float(ticker['last'])
                    amount = self.calculate_position_size(price)
                    
                    if amount * price <= self.portfolio['USD']:
                        order = self.execute_trade(pair, amount, 'buy')
                        if order:
                            with self.lock:
                                self.portfolio['positions'][pair] = order
                                self.portfolio['USD'] -= order['quantity'] * order['price']
                            
                            msg = (f"Opened {pair}\n"
                                  f"Amount: {order['quantity']:.4f}\n"
                                  f"Price: ${order['price']:.2f}\n"
                                  f"Score: {score:.2f}")
                            self.send_alert(msg)

    def run(self):
        """Main trading loop"""
        self.send_alert("🚀 Trading bot started successfully")
        last_report = datetime.now()
        
        while True:
            try:
                start_time = time.time()
                
                # Rebalance portfolio
                self.rebalance_portfolio()
                
                # Periodic reporting
                if (datetime.now() - last_report) > timedelta(hours=6):
                    total_value = self.portfolio['USD']
                    for pos in self.portfolio['positions'].values():
                        ticker = self.rate_limited_request(exchange.fetch_ticker, pos['pair'])
                        total_value += pos['quantity'] * self.safe_float(ticker['last'])
                    
                    profit_pct = (total_value - INITIAL_BALANCE)/INITIAL_BALANCE*100
                    report = (f"📊 Portfolio Report\n"
                            f"Value: ${total_value:.2f}\n"
                            f"Profit: {profit_pct:.2f}%\n"
                            f"Positions: {len(self.portfolio['positions'])}/{MAX_POSITIONS}")
                    self.send_alert(report)
                    last_report = datetime.now()
                
                # Adaptive sleep
                elapsed = time.time() - start_time
                sleep_time = max(CHECK_INTERVAL - elapsed, 60)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.send_alert("🛑 Bot stopped by user")
                break
            except Exception as e:
                self.send_alert(f"⚠️ Critical error: {str(e)}")
                time.sleep(300)

if __name__ == "__main__":
    trader = KrakenMomentumTrader()
    trader.run()