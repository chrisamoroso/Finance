"""
Stock Screener - Tech Stocks Financial Metrics Calculator

This script pulls financial data for tech stocks and calculates:
- Revenue growth (Q/Q)
- Revenue growth acceleration
- Comp-adjusted acceleration
- Gross margin
- Q/Q gross margin expansion
- Magic number
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Comprehensive list of tech stocks
TECH_STOCKS = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    # Cloud & SaaS
    'CRM', 'NOW', 'SNOW', 'DDOG', 'MDB', 'NET', 'CFLT', 'TEAM', 'WDAY', 'ZS',
    'OKTA', 'CRWD', 'S', 'TWLO', 'ZM', 'DOCU', 'BILL', 'HUBS', 'ASAN', 'MNDY',
    # Semiconductors
    'AMD', 'INTC', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'NXPI', 'ADI',
    'MCHP', 'ON', 'MPWR', 'SWKS', 'QRVO',
    # Software & Enterprise
    'INTU', 'PANW', 'PLTR', 'FTNT', 'SNPS', 'CDNS', 'ANSS', 'TTD', 'VEEV', 'DXCM',
    # E-commerce & Digital
    'SHOP', 'SQ', 'PYPL', 'UBER', 'DASH', 'ABNB', 'SPOT', 'PINS', 'SNAP', 'RBLX',
    # Other Tech
    'IBM', 'CSCO', 'ACN', 'NFLX', 'ADSK', 'HPE', 'HPQ', 'DELL', 'CRM', 'VMW'
]


class StockScreener:
    """Stock screener that calculates growth and profitability metrics"""

    def __init__(self, ticker_list):
        self.tickers = ticker_list
        self.results = []

    def get_quarterly_financials(self, ticker):
        """
        Fetch quarterly financial statements for a given ticker
        Returns income statement and cash flow data
        """
        try:
            stock = yf.Ticker(ticker)

            # Get quarterly financials
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet

            if income_stmt.empty:
                print(f"No data available for {ticker}")
                return None

            return {
                'ticker': ticker,
                'income_stmt': income_stmt,
                'balance_sheet': balance_sheet
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def calculate_revenue_growth(self, revenues):
        """
        Calculate quarter-over-quarter revenue growth rate
        Formula: (Current Q Revenue - Prior Q Revenue) / Prior Q Revenue
        """
        if len(revenues) < 2:
            return [None]

        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i+1] != 0 and not pd.isna(revenues[i+1]):
                growth = (revenues[i] - revenues[i+1]) / revenues[i+1]
                growth_rates.append(growth)
            else:
                growth_rates.append(None)

        return growth_rates

    def calculate_revenue_acceleration(self, growth_rates):
        """
        Calculate revenue growth acceleration
        Formula: Current Growth Rate - Prior Growth Rate
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
        Formula: Current Acceleration - Prior Period Acceleration
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

    def calculate_gross_margin(self, revenues, cogs):
        """
        Calculate gross margin
        Formula: (Revenue - COGS) / Revenue
        """
        gross_margins = []
        for rev, cost in zip(revenues, cogs):
            if rev != 0 and not pd.isna(rev) and not pd.isna(cost):
                margin = (rev - cost) / rev
                gross_margins.append(margin)
            else:
                gross_margins.append(None)

        return gross_margins

    def calculate_margin_expansion(self, margins):
        """
        Calculate quarter-over-quarter gross margin expansion
        Formula: Current Q Margin - Prior Q Margin
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
        Formula: (Current Q Revenue - Prior Q Revenue) / Prior Q Sales & Marketing Expense

        Magic number > 1 is considered excellent for SaaS companies
        Magic number > 0.75 is good
        """
        if len(revenues) < 2 or len(sales_marketing) < 2:
            return [None]

        magic_numbers = []
        for i in range(len(revenues) - 1):
            if (revenues[i+1] != 0 and sales_marketing[i+1] != 0 and
                not pd.isna(revenues[i]) and not pd.isna(revenues[i+1]) and
                not pd.isna(sales_marketing[i+1])):

                net_new_revenue = revenues[i] - revenues[i+1]
                magic_num = net_new_revenue / sales_marketing[i+1]
                magic_numbers.append(magic_num)
            else:
                magic_numbers.append(None)

        return magic_numbers

    def extract_financial_data(self, financial_data):
        """
        Extract relevant financial metrics from the financial statements
        """
        if financial_data is None:
            return None

        ticker = financial_data['ticker']
        income_stmt = financial_data['income_stmt']

        try:
            # Extract revenue (Total Revenue or Operating Revenue)
            if 'Total Revenue' in income_stmt.index:
                revenues = income_stmt.loc['Total Revenue'].values
            elif 'Operating Revenue' in income_stmt.index:
                revenues = income_stmt.loc['Operating Revenue'].values
            else:
                print(f"Revenue data not found for {ticker}")
                return None

            # Extract Cost of Revenue (COGS)
            if 'Cost Of Revenue' in income_stmt.index:
                cogs = income_stmt.loc['Cost Of Revenue'].values
            else:
                print(f"COGS data not found for {ticker}")
                cogs = np.zeros(len(revenues))

            # Extract Sales & Marketing (if available)
            if 'Selling General And Administration' in income_stmt.index:
                sales_marketing = income_stmt.loc['Selling General And Administration'].values
            else:
                # Use operating expenses as proxy if SG&A not available
                if 'Operating Expense' in income_stmt.index:
                    sales_marketing = income_stmt.loc['Operating Expense'].values
                else:
                    sales_marketing = np.zeros(len(revenues))

            return {
                'ticker': ticker,
                'dates': income_stmt.columns.tolist(),
                'revenues': revenues.tolist(),
                'cogs': cogs.tolist(),
                'sales_marketing': sales_marketing.tolist()
            }

        except Exception as e:
            print(f"Error extracting data for {ticker}: {str(e)}")
            return None

    def calculate_all_metrics(self, data):
        """
        Calculate all metrics for a given stock
        """
        if data is None:
            return None

        revenues = data['revenues']
        cogs = data['cogs']
        sales_marketing = data['sales_marketing']

        # Calculate all metrics
        revenue_growth = self.calculate_revenue_growth(revenues)
        revenue_acceleration = self.calculate_revenue_acceleration(revenue_growth)
        comp_adj_acceleration = self.calculate_comp_adjusted_acceleration(revenue_acceleration)
        gross_margins = self.calculate_gross_margin(revenues, cogs)
        margin_expansion = self.calculate_margin_expansion(gross_margins)
        magic_numbers = self.calculate_magic_number(revenues, sales_marketing)

        # Get most recent metrics (index 0 is most recent)
        result = {
            'Ticker': data['ticker'],
            'Latest Quarter': data['dates'][0].strftime('%Y-%m-%d') if len(data['dates']) > 0 else None,
            'Revenue (Latest)': revenues[0] if len(revenues) > 0 else None,
            'Revenue Growth (Q/Q)': revenue_growth[0] if len(revenue_growth) > 0 and revenue_growth[0] is not None else None,
            'Revenue Acceleration': revenue_acceleration[0] if len(revenue_acceleration) > 0 and revenue_acceleration[0] is not None else None,
            'Comp-Adjusted Acceleration': comp_adj_acceleration[0] if len(comp_adj_acceleration) > 0 and comp_adj_acceleration[0] is not None else None,
            'Gross Margin': gross_margins[0] if len(gross_margins) > 0 and gross_margins[0] is not None else None,
            'Gross Margin Expansion (Q/Q)': margin_expansion[0] if len(margin_expansion) > 0 and margin_expansion[0] is not None else None,
            'Magic Number': magic_numbers[0] if len(magic_numbers) > 0 and magic_numbers[0] is not None else None,
        }

        return result

    def screen_stocks(self):
        """
        Main method to screen all stocks and calculate metrics
        """
        print(f"Screening {len(self.tickers)} tech stocks...")
        print("=" * 80)

        for ticker in self.tickers:
            print(f"\nProcessing {ticker}...")

            # Get financial data
            financial_data = self.get_quarterly_financials(ticker)

            # Extract relevant metrics
            extracted_data = self.extract_financial_data(financial_data)

            # Calculate all metrics
            metrics = self.calculate_all_metrics(extracted_data)

            if metrics:
                self.results.append(metrics)
                print(f"✓ {ticker} processed successfully")
            else:
                print(f"✗ {ticker} - insufficient data")

        print("\n" + "=" * 80)
        print(f"Completed screening {len(self.results)} stocks")

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Sort by revenue growth (descending)
        if not df.empty and 'Revenue Growth (Q/Q)' in df.columns:
            df = df.sort_values('Revenue Growth (Q/Q)', ascending=False)

        return df

    def save_results(self, df, filename='stock_screener_results.csv'):
        """
        Save results to CSV file
        """
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

        # Also save to Excel for easier viewing
        excel_filename = filename.replace('.csv', '.xlsx')
        df.to_excel(excel_filename, index=False)
        print(f"Results saved to {excel_filename}")


def main():
    """
    Main execution function
    """
    print("Tech Stock Screener - Financial Metrics Calculator")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of stocks to analyze: {len(TECH_STOCKS)}")
    print("=" * 80)

    # Initialize screener
    screener = StockScreener(TECH_STOCKS)

    # Run screening
    results_df = screener.screen_stocks()

    # Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal stocks analyzed: {len(results_df)}")

    if not results_df.empty:
        print("\nTop 10 by Revenue Growth:")
        print(results_df[['Ticker', 'Revenue Growth (Q/Q)', 'Gross Margin', 'Magic Number']].head(10).to_string())

        print("\n\nMetrics Overview:")
        print(results_df.describe())

    # Save results
    screener.save_results(results_df)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    results = main()
