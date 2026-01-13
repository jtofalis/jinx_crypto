#!/usr/bin/env python3
"""
Comprehensive Trading Performance Analyzer
Analyzes autotrader_trades.csv and generates detailed reports
"""

import csv
import json
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import math

def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        # Handle MANUAL or other string entries
        if isinstance(value, str) and value.upper() == 'MANUAL':
            return default
        return float(value) if value else default
    except (ValueError, TypeError):
        return default

def safe_datetime(date_str):
    """Safely parse datetime"""
    if not date_str:
        return None
    try:
        # Handle various datetime formats
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        return datetime.fromisoformat(date_str)
    except Exception as e:
        print(f"⚠️  Warning: Could not parse datetime '{date_str}': {e}")
        return None

def load_trades(trades_csv_path):
    """Load trades from CSV file"""
    
    if not os.path.exists(trades_csv_path):
        print(f"❌ Trade file not found: {trades_csv_path}")
        print(f"   Expected location: {os.path.abspath(trades_csv_path)}")
        return None
    
    trades = []
    with open(trades_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    
    if not trades:
        print("❌ No trades found in CSV")
        return None
    
    print(f"✅ Loaded {len(trades)} trade records from {trades_csv_path}\n")
    return trades

def match_trades(trades):
    """Match buy and sell orders to create completed trades using FIFO method"""
    
    buys = [t for t in trades if t.get('side') == 'BUY']
    sells = [t for t in trades if t.get('side') == 'SELL']
    
    print(f"📊 Found {len(buys)} BUY orders and {len(sells)} SELL orders")
    
    matched_trades = []
    
    # Group trades by symbol and use FIFO matching
    symbols = set(t.get('symbol') for t in trades if t.get('symbol'))
    
    for symbol in symbols:
        # Get buys and sells for this symbol, sorted by time
        symbol_buys = sorted(
            [b for b in buys if b.get('symbol') == symbol],
            key=lambda x: safe_datetime(x.get('time')) or datetime.min
        )
        symbol_sells = sorted(
            [s for s in sells if s.get('symbol') == symbol],
            key=lambda x: safe_datetime(x.get('time')) or datetime.min
        )
        
        # FIFO matching: match each sell with available buy inventory
        buy_inventory = []  # List of (buy_record, remaining_amount)
        
        for buy in symbol_buys:
            amount = safe_float(buy.get('amount'))
            if amount > 0:
                buy_inventory.append({'record': buy, 'remaining': amount})
        
        for sell in symbol_sells:
            sell_amount = safe_float(sell.get('amount'))
            sell_time = safe_datetime(sell.get('time'))
            
            if sell_amount <= 0:
                continue
            
            # Match with inventory FIFO
            remaining_sell = sell_amount
            
            while remaining_sell > 0.00000001 and buy_inventory:
                oldest_buy = buy_inventory[0]
                buy_record = oldest_buy['record']
                buy_remaining = oldest_buy['remaining']
                buy_time = safe_datetime(buy_record.get('time'))
                
                # Only match if buy came before sell
                if buy_time and sell_time and buy_time >= sell_time:
                    break
                
                # Match the amounts
                matched_amount = min(remaining_sell, buy_remaining)
                
                # Calculate P&L for this matched portion
                entry_price = safe_float(buy_record.get('price'))
                
                # Handle MANUAL price entries - when price is MANUAL, actual price is in amount field
                price_value = sell.get('price')
                if isinstance(price_value, str) and price_value.upper() == 'MANUAL':
                    exit_price = safe_float(sell.get('amount'))  # Actual price is in amount field
                    actual_sell_amount = safe_float(sell.get('notional_usd'))  # Actual amount is in notional field
                    recorded_pnl = safe_float(sell.get('dry_run'))  # Actual PnL is in dry_run field
                else:
                    exit_price = safe_float(sell.get('price'))
                    actual_sell_amount = sell_amount
                    recorded_pnl = safe_float(sell.get('pnl_usd'))
                
                # Calculate proportion of this sell that's being matched
                sell_proportion = matched_amount / actual_sell_amount if actual_sell_amount > 0 else 1.0
                
                # Allocate PnL proportionally
                if recorded_pnl != 0:
                    pnl = recorded_pnl * sell_proportion
                else:
                    pnl = (exit_price - entry_price) * matched_amount
                
                # Only calculate percentage if we have valid prices
                if entry_price > 0 and exit_price > 0:
                    pnl_pct = ((exit_price - entry_price) / entry_price * 100)
                else:
                    # If we don't have valid prices but have PnL, approximate from PnL
                    if recorded_pnl != 0 and entry_price > 0:
                        pnl_pct = (pnl / (entry_price * matched_amount)) * 100
                    else:
                        pnl_pct = 0
                
                # Calculate duration
                duration_hours = ((sell_time - buy_time).total_seconds() / 3600) if buy_time and sell_time else 0
                
                # Notional value
                notional = safe_float(buy_record.get('notional_usd')) or (entry_price * matched_amount)
                
                matched_trades.append({
                    'symbol': symbol,
                    'entry_time': buy_record.get('time'),
                    'exit_time': sell.get('time'),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'amount': matched_amount,
                    'notional': notional,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_hours': duration_hours,
                    'exit_reason': sell.get('reason', 'UNKNOWN'),
                    'entry_reason': buy_record.get('reason', 'UNKNOWN'),
                    'dry_run': buy_record.get('dry_run', 'true').lower() in ('true', '1', 'yes'),
                    'buy_record': buy_record,
                    'sell_record': sell
                })
                
                # Update remaining amounts
                remaining_sell -= matched_amount
                buy_inventory[0]['remaining'] -= matched_amount
                
                # Remove buy from inventory if fully consumed
                if buy_inventory[0]['remaining'] < 0.00000001:
                    buy_inventory.pop(0)
    
    # Find unmatched buys (still in inventory)
    unmatched_buys = []
    for symbol in symbols:
        symbol_buys = sorted(
            [b for b in buys if b.get('symbol') == symbol],
            key=lambda x: safe_datetime(x.get('time')) or datetime.min
        )
        symbol_sells = sorted(
            [s for s in sells if s.get('symbol') == symbol],
            key=lambda x: safe_datetime(x.get('time')) or datetime.min
        )
        
        buy_inventory = []
        for buy in symbol_buys:
            amount = safe_float(buy.get('amount'))
            if amount > 0:
                buy_inventory.append({'record': buy, 'remaining': amount})
        
        for sell in symbol_sells:
            sell_amount = safe_float(sell.get('amount'))
            sell_time = safe_datetime(sell.get('time'))
            remaining_sell = sell_amount
            
            while remaining_sell > 0.00000001 and buy_inventory:
                oldest_buy = buy_inventory[0]
                buy_time = safe_datetime(oldest_buy['record'].get('time'))
                
                if buy_time and sell_time and buy_time >= sell_time:
                    break
                
                matched_amount = min(remaining_sell, oldest_buy['remaining'])
                remaining_sell -= matched_amount
                buy_inventory[0]['remaining'] -= matched_amount
                
                if buy_inventory[0]['remaining'] < 0.00000001:
                    buy_inventory.pop(0)
        
        # Remaining inventory are unmatched buys
        for item in buy_inventory:
            if item['remaining'] > 0.00000001:
                unmatched_buys.append(item['record'])
    
    print(f"✅ Matched {len(matched_trades)} completed trades")
    if unmatched_buys:
        print(f"⚠️  {len(unmatched_buys)} open positions (no matching sell)")
    
    return {
        'matched_trades': matched_trades,
        'unmatched_buys': unmatched_buys,
        'all_sells': sells
    }

def calculate_statistics(matched_trades):
    """Calculate comprehensive trading statistics"""
    
    if not matched_trades:
        return None
    
    total_trades = len(matched_trades)
    
    # Separate winning and losing trades
    winning_trades = [t for t in matched_trades if t['pnl'] > 0]
    losing_trades = [t for t in matched_trades if t['pnl'] <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    # P&L calculations
    total_pnl = sum(t['pnl'] for t in matched_trades)
    total_win_pnl = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
    total_loss_pnl = sum(t['pnl'] for t in losing_trades) if losing_trades else 0
    
    avg_pnl = total_pnl / total_trades if total_trades else 0
    avg_win = total_win_pnl / win_count if win_count else 0
    avg_loss = total_loss_pnl / loss_count if loss_count else 0
    
    # Win rate
    win_rate = (win_count / total_trades * 100) if total_trades else 0
    
    # Risk/Reward ratio
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Profit factor
    profit_factor = abs(total_win_pnl / total_loss_pnl) if total_loss_pnl != 0 else float('inf')
    
    # Expectancy (average $ per trade)
    expectancy = avg_pnl
    
    # Duration statistics
    avg_duration = sum(t['duration_hours'] for t in matched_trades) / total_trades if total_trades else 0
    max_duration = max(t['duration_hours'] for t in matched_trades) if matched_trades else 0
    min_duration = min(t['duration_hours'] for t in matched_trades) if matched_trades else 0
    
    # Consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for t in matched_trades:
        if t['pnl'] > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    # Largest win/loss
    largest_win = max(winning_trades, key=lambda x: x['pnl']) if winning_trades else None
    largest_loss = min(losing_trades, key=lambda x: x['pnl']) if losing_trades else None
    
    return {
        'total_trades': total_trades,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_win_pnl': total_win_pnl,
        'total_loss_pnl': total_loss_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': rr_ratio,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_duration': avg_duration,
        'max_duration': max_duration,
        'min_duration': min_duration,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'largest_win': largest_win,
        'largest_loss': largest_loss
    }

def print_report(stats, matched_trades, unmatched_buys):
    """Print comprehensive performance report"""
    
    if not stats:
        print("❌ No statistics to display")
        return
    
    print("\n" + "=" * 70)
    print("📊 TRADING PERFORMANCE REPORT".center(70))
    print("=" * 70)
    print()
    
    # Overall Statistics
    print("📈 OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total Completed Trades:          {stats['total_trades']:>10,}")
    print(f"Winning Trades:                  {stats['win_count']:>10,}  ({stats['win_rate']:.1f}%)")
    print(f"Losing Trades:                   {stats['loss_count']:>10,}  ({100-stats['win_rate']:.1f}%)")
    print(f"Max Consecutive Wins:            {stats['max_consecutive_wins']:>10,}")
    print(f"Max Consecutive Losses:          {stats['max_consecutive_losses']:>10,}")
    print()
    
    # Profit & Loss
    print("💰 PROFIT & LOSS")
    print("-" * 70)
    pnl_color = "+" if stats['total_pnl'] >= 0 else ""
    print(f"Total P&L:                       {pnl_color}${stats['total_pnl']:>10,.2f}")
    print(f"Average P&L per Trade:           {pnl_color}${stats['avg_pnl']:>10,.2f}")
    print(f"Expected Value per Trade:        {pnl_color}${stats['expectancy']:>10,.2f}")
    print()
    print(f"Total Winning P&L:               +${stats['total_win_pnl']:>10,.2f}")
    print(f"Average Win:                     +${stats['avg_win']:>10,.2f}")
    print()
    print(f"Total Losing P&L:                -${abs(stats['total_loss_pnl']):>10,.2f}")
    print(f"Average Loss:                    -${abs(stats['avg_loss']):>10,.2f}")
    print()
    
    # Performance Metrics
    print("📊 PERFORMANCE METRICS")
    print("-" * 70)
    print(f"Win Rate:                        {stats['win_rate']:>10.1f}%")
    print(f"Risk/Reward Ratio:               {stats['rr_ratio']:>10.2f}:1")
    
    pf_display = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "∞"
    print(f"Profit Factor:                   {pf_display:>10s}")
    print()
    print(f"Average Hold Time:               {stats['avg_duration']:>10.1f} hours")
    print(f"Longest Trade:                   {stats['max_duration']:>10.1f} hours")
    print(f"Shortest Trade:                  {stats['min_duration']:>10.1f} hours")
    print()
    
    # Exit Reason Analysis
    print("🎯 EXIT REASON BREAKDOWN")
    print("-" * 70)
    exit_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    
    for t in matched_trades:
        reason = t['exit_reason']
        exit_stats[reason]['count'] += 1
        exit_stats[reason]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            exit_stats[reason]['wins'] += 1
    
    for reason in sorted(exit_stats.keys()):
        data = exit_stats[reason]
        count = data['count']
        pnl = data['pnl']
        wr = (data['wins'] / count * 100) if count else 0
        pct = (count / stats['total_trades'] * 100) if stats['total_trades'] else 0
        
        pnl_sign = "+" if pnl >= 0 else ""
        print(f"{reason:20s} {count:>4} trades ({pct:>5.1f}%)  "
              f"WR: {wr:>5.1f}%  P&L: {pnl_sign}${pnl:>8.2f}")
    print()
    
    # Symbol Performance
    print("📈 SYMBOL PERFORMANCE")
    print("-" * 70)
    symbol_stats = defaultdict(lambda: {'trades': [], 'pnl': 0, 'wins': 0})
    
    for t in matched_trades:
        symbol = t['symbol']
        symbol_stats[symbol]['trades'].append(t)
        symbol_stats[symbol]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            symbol_stats[symbol]['wins'] += 1
    
    # Sort by P&L
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
    
    for symbol, data in sorted_symbols:
        count = len(data['trades'])
        pnl = data['pnl']
        wr = (data['wins'] / count * 100) if count else 0
        pnl_sign = "+" if pnl >= 0 else ""
        
        print(f"{symbol:15s} {count:>4} trades  WR: {wr:>5.1f}%  "
              f"P&L: {pnl_sign}${pnl:>8.2f}")
    print()
    
    # Best Trades
    print("🏆 TOP 5 BEST TRADES")
    print("-" * 70)
    best_trades = sorted(matched_trades, key=lambda x: x['pnl'], reverse=True)[:5]
    
    for i, t in enumerate(best_trades, 1):
        print(f"{i}. {t['symbol']:12s} +${t['pnl']:>7.2f} ({t['pnl_pct']:>+6.2f}%)  "
              f"[{t['exit_reason']}]  {safe_datetime(t['exit_time']).strftime('%Y-%m-%d %H:%M') if safe_datetime(t['exit_time']) else 'N/A'}")
    print()
    
    # Worst Trades
    print("💔 TOP 5 WORST TRADES")
    print("-" * 70)
    worst_trades = sorted(matched_trades, key=lambda x: x['pnl'])[:5]
    
    for i, t in enumerate(worst_trades, 1):
        print(f"{i}. {t['symbol']:12s} -${abs(t['pnl']):>7.2f} ({t['pnl_pct']:>+6.2f}%)  "
              f"[{t['exit_reason']}]  {safe_datetime(t['exit_time']).strftime('%Y-%m-%d %H:%M') if safe_datetime(t['exit_time']) else 'N/A'}")
    print()
    
    # Open Positions
    if unmatched_buys:
        print("⚠️  OPEN POSITIONS (NO MATCHING SELL)")
        print("-" * 70)
        for buy in unmatched_buys:
            symbol = buy.get('symbol')
            amount = safe_float(buy.get('amount'))
            price = safe_float(buy.get('price'))
            notional = safe_float(buy.get('notional_usd')) or (amount * price)
            time = safe_datetime(buy.get('time'))
            time_str = time.strftime('%Y-%m-%d %H:%M') if time else 'Unknown'
            
            print(f"{symbol:15s} {amount:>12.8f} @ ${price:>10.2f}  "
                  f"Notional: ${notional:>8.2f}  {time_str}")
        print()
    
    print("=" * 70)
    print()

def export_json_report(stats, matched_trades, unmatched_buys, output_path):
    """Export detailed JSON report"""
    
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'statistics': stats,
        'completed_trades': matched_trades,
        'open_positions': [
            {
                'symbol': b.get('symbol'),
                'amount': safe_float(b.get('amount')),
                'entry_price': safe_float(b.get('price')),
                'entry_time': b.get('time'),
                'notional': safe_float(b.get('notional_usd'))
            }
            for b in unmatched_buys
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Detailed JSON report exported to: {output_path}")

def main():
    """Main execution function"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze trading performance from autotrader_trades.csv'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        help='Path to the trades CSV file (default: autotrader_trades.csv)'
    )
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=None,
        help='Lookback period in days (e.g., -d 7 for last 7 days)'
    )
    
    args = parser.parse_args()
    
    # Determine CSV path
    if args.csv_file:
        trades_csv = args.csv_file
    else:
        # Try common locations
        possible_paths = [
            'autotrader_trades.csv',
            './autotrader_trades.csv',
            '../autotrader_trades.csv',
            os.path.expanduser('~/autotrader_trades.csv')
        ]
        
        trades_csv = None
        for path in possible_paths:
            if os.path.exists(path):
                trades_csv = path
                break
        
        if not trades_csv:
            print("❌ Could not find autotrader_trades.csv")
            print("   Please specify path: python analyze_trades.py <path_to_csv>")
            print("   Or use -d flag for lookback period: python analyze_trades.py -d 7")
            return
    
    # Load trades
    trades = load_trades(trades_csv)
    if not trades:
        return
    
    # Filter by lookback period if specified
    if args.days is not None:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=args.days)
        original_count = len(trades)
        trades = [
            t for t in trades
            if safe_datetime(t.get('time')) and safe_datetime(t.get('time')) >= cutoff_date
        ]
        filtered_count = len(trades)
        print(f"📅 Filtering to last {args.days} days: {filtered_count}/{original_count} trades\n")
        
        if not trades:
            print(f"❌ No trades found in the last {args.days} days")
            return
    
    # Match trades
    data = match_trades(trades)
    if not data or not data['matched_trades']:
        print("❌ No completed trades found")
        return
    
    # Calculate statistics
    stats = calculate_statistics(data['matched_trades'])
    if not stats:
        return
    
    # Print report
    print_report(stats, data['matched_trades'], data['unmatched_buys'])
    
    # Export JSON report
    json_output = trades_csv.replace('.csv', '_analysis.json')
    export_json_report(stats, data['matched_trades'], data['unmatched_buys'], json_output)

if __name__ == '__main__':
    main()
