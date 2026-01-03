"""
Quick test script for the stock screener
Tests with a small sample of stocks
"""

from stock_screener import StockScreener
import pandas as pd

# Test with a small sample of well-known tech stocks
TEST_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']

print("Testing Stock Screener with sample stocks...")
print("=" * 80)

# Initialize screener with test stocks
screener = StockScreener(TEST_STOCKS)

# Run screening
results_df = screener.screen_stocks()

# Display results
print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

if not results_df.empty:
    print("\nAll metrics for test stocks:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(results_df.to_string())

    # Save test results
    results_df.to_csv('test_screener_results.csv', index=False)
    print("\n\nTest results saved to test_screener_results.csv")
else:
    print("\nNo results generated - check for errors above")

print("=" * 80)
