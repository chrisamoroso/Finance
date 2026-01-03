"""
Stock Screener - Tech Stocks Financial Metrics Calculator (API Version)

This version uses Financial Modeling Prep API (free tier available)
to fetch financial data and calculate:
- Revenue growth (Q/Q)
- Revenue growth acceleration
- Comp-adjusted acceleration
- Gross margin
- Q/Q gross margin expansion
- Magic number

Get your free API key at: https://financialmodelingprep.com/developer/docs/
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time


# Comprehensive list of tech stocks
TECH_STOCKS = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    # Cloud & SaaS
    'CRM', 'NOW', 'SNOW', 'DDOG', 'MDB', 'NET', 'TEAM', 'WDAY', 'ZS',
    'OKTA', 'CRWD', 'TWLO', 'ZM', 'DOCU', 'BILL', 'HUBS', 'ASAN', 'MNDY',
    # Semiconductors
    'AMD', 'INTC', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'NXPI', 'ADI',
    # Software & Enterprise
    'INTU', 'PANW', 'PLTR', 'FTNT', 'SNPS', 'CDNS', 'ANSS', 'TTD', 'VEEV', 'DXCM',
    # E-commerce & Digital
    'SHOP', 'SQ', 'PYPL', 'UBER', 'DASH', 'ABNB', 'SPOT', 'RBLX',
]

# You can get a free API key from https://financialmodelingprep.com/developer/docs/
# Free tier allows 250 requests per day
FMP_API_KEY = "demo"  # Replace with your actual API key for more requests


class StockScreenerAPI:
    """Stock screener using Financial Modeling Prep API"""

    def __init__(self, ticker_list, api_key=FMP_API_KEY):
        self.tickers = ticker_list
        self.api_key = api_key
        self.results = []
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_income_statement(self, ticker, period='quarter', limit=8):
        """
        Fetch income statement data from FMP API
        """
        url = f"{self.base_url}/income-statement/{ticker}"
        params = {
            'period': period,
            'limit': limit,
            'apikey': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"Error fetching data for {ticker}: Status {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching data for {ticker}: {str(e)}")
            return None

    def calculate_revenue_growth(self, revenues):
        """
        Calculate quarter-over-quarter revenue growth rate
        """
        if len(revenues) < 2:
            return [None]

        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i+1] != 0 and not pd.isna(revenues[i+1]):
                growth = (revenues[i] - revenues[i+1]) / abs(revenues[i+1])
                growth_rates.append(growth)
            else:
                growth_rates.append(None)

        return growth_rates

    def calculate_revenue_acceleration(self, growth_rates):
        """
        Calculate revenue growth acceleration
        """
        if len(growth_rates) < 2:
            return [None]

        acceleration = []
        for i in range(len(growth_rates) - 1):
            if growth_rates[i] is not None and growth_rates[i+1] is not None:
                accel = growth_rates[i] - growth_rates[i+1]
                acceleration.append(accel)
            else:
                acceleration.append(None)

        return acceleration

    def calculate_comp_adjusted_acceleration(self, acceleration):
        """
        Calculate comp-adjusted acceleration
        """
        if len(acceleration) < 2:
            return [None]

        comp_adj_accel = []
        for i in range(len(acceleration) - 1):
            if acceleration[i] is not None and acceleration[i+1] is not None:
                adj_accel = acceleration[i] - acceleration[i+1]
                comp_adj_accel.append(adj_accel)
            else:
                comp_adj_accel.append(None)

        return comp_adj_accel

    def calculate_gross_margin(self, revenues, gross_profits):
        """
        Calculate gross margin
        """
        gross_margins = []
        for rev, gp in zip(revenues, gross_profits):
            if rev != 0 and not pd.isna(rev) and not pd.isna(gp):
                margin = gp / rev
                gross_margins.append(margin)
            else:
                gross_margins.append(None)

        return gross_margins

    def calculate_margin_expansion(self, margins):
        """
        Calculate quarter-over-quarter gross margin expansion
        """
        if len(margins) < 2:
            return [None]

        expansion = []
        for i in range(len(margins) - 1):
            if margins[i] is not None and margins[i+1] is not None:
                exp = margins[i] - margins[i+1]
                expansion.append(exp)
            else:
                expansion.append(None)

        return expansion

    def calculate_magic_number(self, revenues, sales_marketing):
        """
        Calculate magic number
        Formula: Quarter-over-Quarter Revenue Growth / Prior Quarter Sales & Marketing
        """
        if len(revenues) < 2 or len(sales_marketing) < 2:
            return [None]

        magic_numbers = []
        for i in range(len(revenues) - 1):
            if (sales_marketing[i+1] != 0 and
                not pd.isna(revenues[i]) and not pd.isna(revenues[i+1]) and
                not pd.isna(sales_marketing[i+1])):

                net_new_revenue = revenues[i] - revenues[i+1]
                magic_num = net_new_revenue / abs(sales_marketing[i+1])
                magic_numbers.append(magic_num)
            else:
                magic_numbers.append(None)

        return magic_numbers

    def process_ticker(self, ticker):
        """
        Process a single ticker and calculate all metrics
        """
        print(f"Processing {ticker}...", end=" ")

        # Get income statement data
        income_data = self.get_income_statement(ticker)

        if not income_data or len(income_data) == 0:
            print(f"✗ No data")
            return None

        try:
            # Extract data from API response
            dates = [item.get('date', '') for item in income_data]
            revenues = [item.get('revenue', 0) for item in income_data]
            gross_profits = [item.get('grossProfit', 0) for item in income_data]
            cogs = [item.get('costOfRevenue', 0) for item in income_data]

            # Sales & Marketing (use sellingGeneralAndAdministrativeExpenses as proxy)
            sales_marketing = [item.get('sellingGeneralAndAdministrativeExpenses', 0) for item in income_data]

            # Calculate all metrics
            revenue_growth = self.calculate_revenue_growth(revenues)
            revenue_acceleration = self.calculate_revenue_acceleration(revenue_growth)
            comp_adj_acceleration = self.calculate_comp_adjusted_acceleration(revenue_acceleration)
            gross_margins = self.calculate_gross_margin(revenues, gross_profits)
            margin_expansion = self.calculate_margin_expansion(gross_margins)
            magic_numbers = self.calculate_magic_number(revenues, sales_marketing)

            # Prepare result dictionary
            result = {
                'Ticker': ticker,
                'Latest Quarter': dates[0] if len(dates) > 0 else None,
                'Revenue (Latest)': revenues[0] if len(revenues) > 0 else None,
                'Revenue Growth (Q/Q)': revenue_growth[0] if len(revenue_growth) > 0 and revenue_growth[0] is not None else None,
                'Revenue Acceleration': revenue_acceleration[0] if len(revenue_acceleration) > 0 and revenue_acceleration[0] is not None else None,
                'Comp-Adjusted Acceleration': comp_adj_acceleration[0] if len(comp_adj_acceleration) > 0 and comp_adj_acceleration[0] is not None else None,
                'Gross Margin': gross_margins[0] if len(gross_margins) > 0 and gross_margins[0] is not None else None,
                'Gross Margin Expansion (Q/Q)': margin_expansion[0] if len(margin_expansion) > 0 and margin_expansion[0] is not None else None,
                'Magic Number': magic_numbers[0] if len(magic_numbers) > 0 and magic_numbers[0] is not None else None,
            }

            print("✓")
            return result

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None

    def screen_stocks(self):
        """
        Screen all stocks and calculate metrics
        """
        print(f"\nScreening {len(self.tickers)} tech stocks...")
        print("=" * 80)

        for i, ticker in enumerate(self.tickers):
            result = self.process_ticker(ticker)

            if result:
                self.results.append(result)

            # Rate limiting - pause between requests
            if (i + 1) % 5 == 0:
                time.sleep(1)  # Pause for 1 second every 5 requests

        print("\n" + "=" * 80)
        print(f"Successfully processed {len(self.results)} stocks")

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Sort by revenue growth
        if not df.empty and 'Revenue Growth (Q/Q)' in df.columns:
            df = df.sort_values('Revenue Growth (Q/Q)', ascending=False)

        return df

    def save_results(self, df, filename='stock_screener_results.csv'):
        """
        Save results to CSV and Excel
        """
        if df.empty:
            print("\nNo results to save")
            return

        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

        try:
            excel_filename = filename.replace('.csv', '.xlsx')
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            print(f"Results saved to {excel_filename}")
        except Exception as e:
            print(f"Could not save Excel file: {str(e)}")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print("Tech Stock Screener - Financial Metrics Calculator (API Version)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of stocks to analyze: {len(TECH_STOCKS)}")
    print(f"API: Financial Modeling Prep (free tier: demo key)")
    print("=" * 80)

    # Initialize screener
    screener = StockScreenerAPI(TECH_STOCKS)

    # Run screening
    results_df = screener.screen_stocks()

    if not results_df.empty:
        # Display summary
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal stocks successfully analyzed: {len(results_df)}")

        # Top performers by revenue growth
        print("\n" + "-" * 80)
        print("Top 10 by Revenue Growth (Q/Q):")
        print("-" * 80)
        top_growth = results_df[['Ticker', 'Revenue Growth (Q/Q)', 'Gross Margin', 'Magic Number']].head(10)
        print(top_growth.to_string(index=False))

        # Best gross margins
        print("\n" + "-" * 80)
        print("Top 10 by Gross Margin:")
        print("-" * 80)
        df_sorted_margin = results_df.sort_values('Gross Margin', ascending=False)
        top_margin = df_sorted_margin[['Ticker', 'Gross Margin', 'Revenue Growth (Q/Q)', 'Gross Margin Expansion (Q/Q)']].head(10)
        print(top_margin.to_string(index=False))

        # Statistical overview
        print("\n" + "-" * 80)
        print("Statistical Overview:")
        print("-" * 80)
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        print(results_df[numeric_cols].describe())

        # Save results
        screener.save_results(results_df)

    else:
        print("\nNo results generated. Check API key and network connection.")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    results = main()
