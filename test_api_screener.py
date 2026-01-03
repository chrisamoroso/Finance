"""
Quick test of the API-based stock screener
"""

from stock_screener_api import StockScreenerAPI
import pandas as pd

# Test with a small sample
TEST_STOCKS = ['AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL']

print("\n" + "=" * 80)
print("Testing Stock Screener (API Version)")
print("=" * 80)

# Initialize and run
screener = StockScreenerAPI(TEST_STOCKS)
results_df = screener.screen_stocks()

if not results_df.empty:
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print("\n", results_df.to_string(index=False))

    # Save test results
    results_df.to_csv('test_api_screener_results.csv', index=False)
    print("\n\nTest results saved to test_api_screener_results.csv")
else:
    print("\n✗ No results - check API connectivity")

print("=" * 80)
