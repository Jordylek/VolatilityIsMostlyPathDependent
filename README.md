# Volatility is (Almost) Path Dependent

Code for the paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4174589.

This code contains the functions to:
- Perform the fitting of one model on historical price (see `historical_analysis.ipynb`). We show one example on SPX and VIX using data downloaded from [NASDAQ](https://www.nasdaq.com/market-activity/index/spx/historical) and [Yahoo! Finance](https://finance.yahoo.com/quote/%5EVIX/history?period1=631238400&period2=1668556800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true). Note that we only have access to the past 10 years of SPX. Therefore, one cannot reproduce our results without finding access to more data of SPX. 
- Compute a SPX or VIX vanilla call price using our 4-factor markovian PDV model (see `option_pricing_4fmpdv.ipynb`)



For any question regarding the code, contact me via email at jordan dot lekeufack at berkeley dot edu.