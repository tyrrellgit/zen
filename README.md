# Zen

> A state of calm attentiveness in which one's actions are guided by intuition rather than by conscious effort

Expose a simple UI for a couple of fairly robust rebalancing portfolios of traditional equities (without trying too hard).

To run this simply install the `requirements.txt` and then run:

`streamlit run main.py` 

and the app will be running at `0.0.0.0:80` (default HTTP address).  If you want to change this just check out the `config.toml` under the `.streamlit` directory.

---

After that, input the stock tickers you want to check out as a portfolio and it will shoot you back a roughly optimised weighting based on some Risk Parity approaches. 
It will also do some basic benchmarking for you against the S&P500 for the same period.  If you can't beat `S&P500 Buy & Hold` then maybe stock picking isn't for you.

I may or may not add some automated stock picking in future.  Who knows.  We aren't try-harding here.