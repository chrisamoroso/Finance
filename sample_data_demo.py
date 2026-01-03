"""
Stock Screener - Sample Data Demonstration

This demonstrates all the metrics calculations with sample data for tech stocks.
This shows exactly how the screener works when connected to real financial APIs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Sample quarterly financial data for demonstration (in millions)
SAMPLE_DATA = {
    'NVDA': {
        'quarters': ['2024-Q3', '2024-Q2', '2024-Q1', '2023-Q4', '2023-Q3', '2023-Q2'],
        'revenue': [35082, 30040, 26044, 22103, 18120, 13507],  # Strong growth
        'gross_profit': [27573, 23715, 20402, 16774, 12902, 9462],
        'sales_marketing': [2662, 2294, 2008, 1838, 1618, 1362]
    },
    'META': {
        'quarters': ['2024-Q3', '2024-Q2', '2024-Q1', '2023-Q4', '2023-Q3', '2023-Q2'],
        'revenue': [40589, 39071, 36455, 40111, 34146, 31999],
        'gross_profit': [32515, 31443, 29328, 31801, 27403, 25492],
        'sales_marketing': [14156, 13715, 13147, 14296, 11597, 10970]
    },
    'MSFT': {
        'quarters': ['2024-Q3', '2024-Q2', '2024-Q1', '2023-Q4', '2023-Q3', '2023-Q2'],
        'revenue': [65585, 64728, 61858, 62020, 56517, 56189],
        'gross_profit': [44945, 44686, 42944, 42731, 38521, 38428],
        'sales_marketing': [6297, 6050, 5831, 6186, 5587, 5590]
    },
    'GOOGL': {
        'quarters': ['2024-Q3', '2024-Q2', '2024-Q1', '2023-Q4', '2023-Q3', '2023-Q2'],
        'revenue': [88268, 84742, 80539, 86310, 76693, 74604],
        'gross_profit': [49396, 47444, 45315, 47635, 42495, 41266],
        'sales_marketing': [10668, 10262, 9766, 10413, 9343, 9088]
    },
    'AAPL': {
        'quarters': ['2024-Q3', '2024-Q2', '2024-Q1', '2023-Q4', '2023-Q3', '2023-Q2'],
        'revenue': [94930, 85777, 90753, 119575, 89498, 81797],
        'gross_profit': [42659, 38495, 41098, 54862, 41137, 36323],
        'sales_marketing': [7188, 6496, 6867, 9050, 6772, 6193]
    },
}


class MetricsCalculator:
    """Calculate all financial metrics"""

    @staticmethod
    def calculate_revenue_growth(revenues):
        """Quarter-over-quarter revenue growth"""
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i+1] != 0:
                growth = (revenues[i] - revenues[i+1]) / revenues[i+1]
                growth_rates.append(growth)
            else:
                growth_rates.append(None)
        return growth_rates

    @staticmethod
    def calculate_acceleration(growth_rates):
        """Change in growth rate (acceleration)"""
        acceleration = []
        for i in range(len(growth_rates) - 1):
            if growth_rates[i] is not None and growth_rates[i+1] is not None:
                accel = growth_rates[i] - growth_rates[i+1]
                acceleration.append(accel)
            else:
                acceleration.append(None)
        return acceleration

    @staticmethod
    def calculate_comp_adjusted_acceleration(acceleration):
        """Comp-adjusted acceleration (second derivative of revenue)"""
        comp_adj = []
        for i in range(len(acceleration) - 1):
            if acceleration[i] is not None and acceleration[i+1] is not None:
                adj = acceleration[i] - acceleration[i+1]
                comp_adj.append(adj)
            else:
                comp_adj.append(None)
        return comp_adj

    @staticmethod
    def calculate_gross_margin(revenues, gross_profits):
        """Gross margin percentage"""
        margins = []
        for rev, gp in zip(revenues, gross_profits):
            if rev != 0:
                margin = gp / rev
                margins.append(margin)
            else:
                margins.append(None)
        return margins

    @staticmethod
    def calculate_margin_expansion(margins):
        """Q/Q change in gross margin"""
        expansion = []
        for i in range(len(margins) - 1):
            if margins[i] is not None and margins[i+1] is not None:
                exp = margins[i] - margins[i+1]
                expansion.append(exp)
            else:
                expansion.append(None)
        return expansion

    @staticmethod
    def calculate_magic_number(revenues, sales_marketing):
        """Magic number: Revenue Growth / Sales & Marketing Spend"""
        magic_numbers = []
        for i in range(len(revenues) - 1):
            if sales_marketing[i+1] != 0:
                revenue_growth = revenues[i] - revenues[i+1]
                magic_num = revenue_growth / sales_marketing[i+1]
                magic_numbers.append(magic_num)
            else:
                magic_numbers.append(None)
        return magic_numbers


def analyze_stock(ticker, data):
    """Analyze a single stock and return metrics"""

    calc = MetricsCalculator()

    revenues = data['revenue']
    gross_profits = data['gross_profit']
    sales_marketing = data['sales_marketing']

    # Calculate all metrics
    revenue_growth = calc.calculate_revenue_growth(revenues)
    acceleration = calc.calculate_acceleration(revenue_growth)
    comp_adj_accel = calc.calculate_comp_adjusted_acceleration(acceleration)
    gross_margins = calc.calculate_gross_margin(revenues, gross_profits)
    margin_expansion = calc.calculate_margin_expansion(gross_margins)
    magic_numbers = calc.calculate_magic_number(revenues, sales_marketing)

    # Return latest metrics
    result = {
        'Ticker': ticker,
        'Latest Quarter': data['quarters'][0],
        'Revenue (M)': revenues[0],
        'Revenue Growth (Q/Q)': revenue_growth[0] if revenue_growth else None,
        'Revenue Acceleration': acceleration[0] if acceleration else None,
        'Comp-Adj Acceleration': comp_adj_accel[0] if comp_adj_accel else None,
        'Gross Margin': gross_margins[0] if gross_margins else None,
        'Margin Expansion (Q/Q)': margin_expansion[0] if margin_expansion else None,
        'Magic Number': magic_numbers[0] if magic_numbers else None,
    }

    return result


def detailed_analysis(ticker, data):
    """Show detailed quarter-by-quarter analysis"""

    calc = MetricsCalculator()

    revenues = data['revenue']
    gross_profits = data['gross_profit']
    sales_marketing = data['sales_marketing']

    revenue_growth = calc.calculate_revenue_growth(revenues)
    acceleration = calc.calculate_acceleration(revenue_growth)
    comp_adj_accel = calc.calculate_comp_adjusted_acceleration(acceleration)
    gross_margins = calc.calculate_gross_margin(revenues, gross_profits)
    margin_expansion = calc.calculate_margin_expansion(gross_margins)
    magic_numbers = calc.calculate_magic_number(revenues, sales_marketing)

    print(f"\n{'='*100}")
    print(f"DETAILED ANALYSIS: {ticker}")
    print(f"{'='*100}")

    # Create detailed DataFrame
    df_data = []
    for i in range(len(revenues)):
        row = {
            'Quarter': data['quarters'][i],
            'Revenue': f"${revenues[i]:,}M",
            'Gross Profit': f"${gross_profits[i]:,}M",
            'Gross Margin': f"{gross_margins[i]:.1%}" if i < len(gross_margins) and gross_margins[i] else "N/A",
            'Revenue Growth': f"{revenue_growth[i]:+.1%}" if i < len(revenue_growth) and revenue_growth[i] else "N/A",
            'Acceleration': f"{acceleration[i]:+.1%}" if i < len(acceleration) and acceleration[i] else "N/A",
            'Magic Number': f"{magic_numbers[i]:.2f}" if i < len(magic_numbers) and magic_numbers[i] else "N/A",
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))

    # Key insights
    print(f"\n{'-'*100}")
    print("KEY METRICS (Latest Quarter):")
    print(f"{'-'*100}")
    if revenue_growth:
        print(f"  Revenue Growth:              {revenue_growth[0]:+.1%}")
    if acceleration:
        print(f"  Revenue Acceleration:        {acceleration[0]:+.1%}")
    if comp_adj_accel:
        print(f"  Comp-Adjusted Acceleration:  {comp_adj_accel[0]:+.1%}")
    if gross_margins:
        print(f"  Gross Margin:                {gross_margins[0]:.1%}")
    if margin_expansion:
        print(f"  Margin Expansion:            {margin_expansion[0]:+.1%}")
    if magic_numbers:
        print(f"  Magic Number:                {magic_numbers[0]:.2f}")
        if magic_numbers[0] > 1:
            print(f"    → Excellent! (>1.0 is very efficient growth)")
        elif magic_numbers[0] > 0.75:
            print(f"    → Good (>0.75 is healthy growth efficiency)")
        else:
            print(f"    → Needs improvement (<0.75)")


def main():
    """Run the demonstration"""

    print("\n" + "="*100)
    print("STOCK SCREENER - SAMPLE DATA DEMONSTRATION")
    print("="*100)
    print("\nThis demonstrates all metric calculations using sample quarterly data.")
    print("All financial figures are in millions (M) of dollars.")
    print("="*100)

    # Analyze all stocks
    results = []
    for ticker, data in SAMPLE_DATA.items():
        result = analyze_stock(ticker, data)
        results.append(result)

    # Create summary DataFrame
    df = pd.DataFrame(results)

    # Sort by revenue growth
    df_sorted = df.sort_values('Revenue Growth (Q/Q)', ascending=False)

    print("\n" + "="*100)
    print("SUMMARY: ALL STOCKS")
    print("="*100)
    print(df_sorted.to_string(index=False))

    # Show top performers
    print("\n" + "="*100)
    print("TOP PERFORMERS BY REVENUE GROWTH")
    print("="*100)
    top_growth = df_sorted[['Ticker', 'Revenue Growth (Q/Q)', 'Revenue Acceleration', 'Magic Number']].head(3)
    for idx, row in top_growth.iterrows():
        print(f"\n{row['Ticker']}:")
        print(f"  Growth: {row['Revenue Growth (Q/Q)']:+.1%}")
        print(f"  Acceleration: {row['Revenue Acceleration']:+.1%}")
        print(f"  Magic Number: {row['Magic Number']:.2f}")

    # Detailed analysis for top stock
    top_ticker = df_sorted.iloc[0]['Ticker']
    detailed_analysis(top_ticker, SAMPLE_DATA[top_ticker])

    # Save results
    output_file = 'sample_screener_results.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"\n{'='*100}")
    print(f"Results saved to: {output_file}")
    print("="*100)

    return df_sorted


if __name__ == "__main__":
    results = main()
    print("\n✓ Sample demonstration complete!")
    print("\nTo use with real data:")
    print("  1. Get a free API key from financialmodelingprep.com")
    print("  2. Update the API_KEY in stock_screener_api.py")
    print("  3. Run: python3 stock_screener_api.py")
    print("\n" + "="*100)
