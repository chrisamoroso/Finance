# Stock Screener - Tech Stocks Financial Metrics

A comprehensive stock screening tool that calculates key financial metrics for technology stocks, including revenue growth, acceleration, margins, and the Magic Number.

## Overview

This stock screener pulls quarterly financial data for tech stocks and calculates:

1. **Revenue Growth (Q/Q)** - Quarter-over-quarter revenue growth rate
2. **Revenue Growth Acceleration** - Change in the growth rate (first derivative)
3. **Comp-Adjusted Acceleration** - Current acceleration minus prior acceleration (second derivative)
4. **Gross Margin** - (Revenue - COGS) / Revenue
5. **Q/Q Gross Margin Expansion** - Quarter-over-quarter change in gross margin
6. **Magic Number** - Net new revenue / Sales & Marketing spend (efficiency metric)

## Files

### Core Scripts

- **`stock_screener_api.py`** - Production version using Financial Modeling Prep API
- **`stock_screener.py`** - Alternative version using yfinance (requires yfinance installation)
- **`sample_data_demo.py`** - Demonstration with sample data (no API required)
- **`test_api_screener.py`** - Test script for API version

### Output Files

- **`stock_screener_results.csv`** - Full screening results
- **`stock_screener_results.xlsx`** - Excel version of results
- **`sample_screener_results.csv`** - Sample demo results

## Metrics Explained

### 1. Revenue Growth (Q/Q)
```
Formula: (Current Quarter Revenue - Prior Quarter Revenue) / Prior Quarter Revenue
```
- Measures quarterly revenue growth rate
- Higher is better for growth stocks
- Tech companies typically target 20%+ annual growth (5%+ quarterly)

### 2. Revenue Growth Acceleration
```
Formula: Current Growth Rate - Prior Growth Rate
```
- Measures if growth is speeding up or slowing down
- Positive = accelerating growth (excellent)
- Negative = decelerating growth (concerning)
- Key indicator of business momentum

### 3. Comp-Adjusted Acceleration
```
Formula: Current Acceleration - Prior Period Acceleration
```
- Second derivative of revenue
- Shows if the rate of acceleration is changing
- Helps identify inflection points in business performance
- Used to compare against prior year comps

### 4. Gross Margin
```
Formula: (Revenue - Cost of Goods Sold) / Revenue
```
- Fundamental profitability metric
- Software/SaaS: 70-90% is typical
- Hardware: 30-50% is typical
- Higher margins = more pricing power and efficiency

### 5. Q/Q Gross Margin Expansion
```
Formula: Current Quarter Margin - Prior Quarter Margin
```
- Measures operating leverage
- Positive = improving unit economics
- Key indicator of scalability

### 6. Magic Number
```
Formula: (Current Q Revenue - Prior Q Revenue) / Prior Q Sales & Marketing Expense
```
- Measures sales efficiency
- \>1.0 = Excellent (generating $1+ of new revenue per $1 of S&M spend)
- \>0.75 = Good
- <0.5 = Concerning (burning cash inefficiently)
- Critical metric for SaaS and high-growth companies

## Installation

### Prerequisites
```bash
pip install pandas numpy requests openpyxl
```

### For yfinance version (optional):
```bash
pip install yfinance
```

Note: yfinance requires multitasking package which may have installation issues on some systems.

## Usage

### Option 1: Using Sample Data (No API Required)

```bash
python3 sample_data_demo.py
```

This demonstrates all calculations with sample data from major tech stocks.

### Option 2: Using Financial Modeling Prep API

1. Get a free API key from [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs/)
   - Free tier: 250 requests/day
   - No credit card required

2. Update the API key in `stock_screener_api.py`:
   ```python
   FMP_API_KEY = "your_api_key_here"
   ```

3. Run the screener:
   ```bash
   python3 stock_screener_api.py
   ```

### Option 3: Using yfinance (if installed)

```bash
python3 stock_screener.py
```

## Customization

### Adding More Stocks

Edit the `TECH_STOCKS` list in any of the screener files:

```python
TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL',  # Add your tickers here
]
```

### Filtering by Sector

Currently focused on tech stocks. To add other sectors:

```python
SAAS_STOCKS = ['CRM', 'NOW', 'SNOW', 'DDOG', 'MDB']
SEMICONDUCTOR_STOCKS = ['NVDA', 'AMD', 'INTC', 'QCOM']
```

### Adjusting Metric Thresholds

Modify the screening criteria in the script:

```python
# Example: Filter for high-growth stocks
high_growth = results_df[results_df['Revenue Growth (Q/Q)'] > 0.10]  # >10% growth

# Example: Filter for efficient growers (high magic number)
efficient_growth = results_df[results_df['Magic Number'] > 1.0]

# Example: Filter for expanding margins
margin_expansion = results_df[results_df['Gross Margin Expansion (Q/Q)'] > 0]
```

## Output Format

The screener generates two files:

1. **CSV** - For data analysis and further processing
2. **Excel** - For easy viewing and manual analysis

### Output Columns

| Column | Description | Example |
|--------|-------------|---------|
| Ticker | Stock symbol | NVDA |
| Latest Quarter | Most recent quarter | 2024-Q3 |
| Revenue (Latest) | Latest quarterly revenue | $35,082M |
| Revenue Growth (Q/Q) | Q/Q growth rate | 16.8% |
| Revenue Acceleration | Change in growth rate | 1.4% |
| Comp-Adjusted Acceleration | Second derivative | 3.9% |
| Gross Margin | Gross profit margin | 78.6% |
| Gross Margin Expansion (Q/Q) | Q/Q margin change | -0.3% |
| Magic Number | Sales efficiency | 2.20 |

## Example Output

```
Top 10 by Revenue Growth:

Ticker  Revenue Growth (Q/Q)  Gross Margin  Magic Number
NVDA              16.8%          78.6%         2.20
AAPL              10.7%          44.9%         1.41
GOOGL              4.2%          55.9%         0.34
META               3.9%          80.1%         0.11
MSFT               1.3%          68.5%         0.14
```

## Interpreting Results

### Strong Growth Profile
- Revenue Growth > 15%
- Positive acceleration
- Gross Margin > 70%
- Magic Number > 1.0

### Healthy but Maturing
- Revenue Growth 5-15%
- Stable or slight deceleration
- Gross Margin 60-70%
- Magic Number 0.75-1.0

### Concerns
- Revenue Growth < 5%
- Negative acceleration for multiple quarters
- Declining gross margins
- Magic Number < 0.5

## Advanced Usage

### Creating Custom Screens

```python
from stock_screener_api import StockScreenerAPI

# Initialize with custom stock list
screener = StockScreenerAPI(['NVDA', 'AMD', 'INTC'])

# Run screening
results = screener.screen_stocks()

# Apply custom filters
high_quality = results[
    (results['Revenue Growth (Q/Q)'] > 0.10) &
    (results['Gross Margin'] > 0.60) &
    (results['Magic Number'] > 0.75)
]

# Save filtered results
high_quality.to_csv('high_quality_stocks.csv', index=False)
```

### Analyzing Trends Over Time

The screener pulls 8 quarters of data. You can modify the code to track metrics over time:

```python
# In the process_ticker method, return all quarters instead of just latest
historical_data = {
    'dates': dates,
    'revenue_growth': revenue_growth,
    'margins': gross_margins,
    'magic_numbers': magic_numbers
}
```

## Technical Details

### Data Sources

1. **Financial Modeling Prep API** (recommended)
   - Comprehensive financial statements
   - Quarterly and annual data
   - Free tier available

2. **yfinance**
   - Free, no API key required
   - Limited to publicly available data
   - May have installation issues

### Calculation Order

1. Extract revenue, COGS, and S&M expense from income statements
2. Calculate revenue growth rates (first derivative)
3. Calculate acceleration (second derivative)
4. Calculate comp-adjusted acceleration (third derivative comparison)
5. Calculate gross margins and margin expansion
6. Calculate magic numbers

### Limitations

- Requires quarterly financial data (not all stocks report)
- S&M expense may not be separately disclosed for all companies
- Magic number calculation uses SG&A as proxy when S&M not available
- API rate limits apply (250 requests/day for free tier)

## Contributing

To add new metrics or features:

1. Add calculation method to the class
2. Update the `process_ticker` method to include new metric
3. Add column to output DataFrame
4. Update documentation

## License

Open source - free to use and modify

## Support

For issues or questions:
- Check API key is valid
- Verify network connectivity
- Ensure pandas, numpy, and requests are installed
- Review error messages for specific tickers

## Future Enhancements

Potential additions:
- [ ] Net dollar retention (NDR) for SaaS companies
- [ ] Rule of 40 (growth rate + profit margin)
- [ ] Cash flow metrics
- [ ] Valuation ratios (P/S, P/E, EV/Sales)
- [ ] Historical trend visualization
- [ ] Automated screening alerts
- [ ] Industry benchmark comparisons

---

**Created**: 2026-01-03
**Version**: 1.0
**Author**: Stock Screener Project
