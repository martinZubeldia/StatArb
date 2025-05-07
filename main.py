import os
# Turn off oneDNN optimizations for deterministic TensorFlow results
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd

# === Statistical & ML imports ===
from sklearn.decomposition import PCA               # extract orthogonal factors from returns
from statsmodels.tsa.stattools import coint         # cointegration test for mean-reversion signals
from tensorflow.keras.models import Sequential      # sequential model API
from tensorflow.keras.layers import Input, LSTM, Dense  # define LSTM architecture
from tensorflow.keras.optimizers import Adam        # adaptive gradient optimizer


defindent = ''
# === DATA HANDLING ===
class DataHandler:
    """
    Loads, sorts, and computes basic metrics from raw market data.

    - Ensures chronological index order for time-series alignment.
    - Computes simple daily percent-change returns for modeling.
    - Optionally stores volume for later impact calculations.
    """
    def __init__(self, price_df: pd.DataFrame, volume_df: pd.DataFrame = None):
        # sort_index() ensures data is in ascending time order
        self.prices = price_df.sort_index()
        # pct_change() calculates daily returns; drop first NaN
        self.returns = self.prices.pct_change().dropna()
        # volumes may be used for impact modeling if provided
        self.volume = volume_df.sort_index() if volume_df is not None else None


# === FEATURE ENGINEERING ===
class FeatureEngineer:
    """
    Creates predictive signals from price & volume data:
      1. PCA components capture broad market movements.
      2. Cointegrated pairs identify mean-reverting spreads.
      3. Momentum measures short-term trend strength.
      4. Volatility estimates risk for position sizing.
      5. Spread z-scores quantify deviation from equilibrium.
    """
    def __init__(self, data: DataHandler):
        self.data = data

    def compute_pca_signals(self, n_components=3) -> pd.DataFrame:
        # fit PCA on returns to extract top factors
        vals = PCA(n_components=n_components).fit_transform(self.data.returns.values)
        cols = [f"PC{i+1}" for i in range(n_components)]
        # align PC time series with original dates
        return pd.DataFrame(vals, index=self.data.returns.index, columns=cols)

    def find_cointegrated_pairs(self, pvalue_threshold=0.2) -> list:
        # detect pairs where price series share a long-run equilibrium
        assets = self.data.prices.columns
        pairs = []
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                a, b = assets[i], assets[j]
                # Engle-Granger: returns (t_stat, p_value, crit_vals)
                _, p_value, _ = coint(self.data.prices[a], self.data.prices[b])
                if p_value < pvalue_threshold:
                    pairs.append((a, b, p_value))
        # sort by strongest cointegration
        return sorted(pairs, key=lambda x: x[2])

    def compute_momentum(self, window=5) -> pd.DataFrame:
        # n-day return as momentum; shift(1) to avoid lookahead bias
        return self.data.prices.pct_change(periods=window).shift(1)

    def compute_volatility(self, window=20) -> pd.DataFrame:
        # rolling std-dev of returns as risk metric; shift(1) for past-only data
        return self.data.returns.rolling(window=window).std().shift(1)

    def compute_spread_zscores(self, pairs: list, window=20) -> pd.DataFrame:
        # build spreads for each pair: price difference
        spreads = pd.DataFrame({
            f"{a}_{b}_spread": self.data.prices[a] - self.data.prices[b]
            for a, b, _ in pairs
        })
        # rolling mean and std to compute z-score
        mu = spreads.rolling(window).mean()
        sigma = spreads.rolling(window).std()
        z = (spreads - mu) / sigma
        # drop initial rows with NaNs
        return z.dropna()


# === LSTM PREDICTION MODEL ===
class LSTMAlphaModel:
    """
    Trains an LSTM network to predict next-day returns for all assets.

    - Input: multi-feature sequences of length `lookback`.
    - Architecture: Input -> LSTM(64 units) -> Dense(n_assets).
    - Loss: MSE between predicted and actual returns.
    """
    def __init__(self, lookback=20, features=None, asset_names=None):
        self.lookback = lookback
        self.features = features or []
        # store asset tickers to build output DataFrame
        self.asset_names = asset_names or []
        self.n_assets = len(self.asset_names)
        self.model = None

    def prepare_sequences(self, feat_df: pd.DataFrame, target_df: pd.DataFrame):
        # convert flat DataFrames into 3D arrays for Keras
        X, Y = [], []
        data_arr = feat_df[self.features].values
        tgt_arr = target_df.values
        for idx in range(self.lookback, len(data_arr)):
            # sequence of past `lookback` steps
            X.append(data_arr[idx-self.lookback:idx])
            # next-day target vector
            Y.append(tgt_arr[idx])
        return np.array(X), np.array(Y)

    def build_model(self):
        # explicit Input layer to satisfy Keras best practices
        self.model = Sequential([
            Input(shape=(self.lookback, len(self.features))),
            LSTM(64, name="lstm_layer"),        # capture temporal dependencies
            Dense(self.n_assets, name="output_layer")  # one output per asset
        ])
        # Adam optimizer balances speed and stability
        self.model.compile(optimizer=Adam(1e-3), loss='mse')

    def train(self, feat_df: pd.DataFrame, target_df: pd.DataFrame,
              epochs=20, batch_size=32):
        # prepare inputs and outputs
        X, Y = self.prepare_sequences(feat_df, target_df)
        self.build_model()
        # verbose=2 prints one line per epoch
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        # create dummy target for correct shaping
        dummy = pd.DataFrame(
            np.zeros((len(feat_df), self.n_assets)),
            index=feat_df.index,
            columns=self.asset_names
        )
        X, _ = self.prepare_sequences(feat_df, dummy)
        preds = self.model.predict(X, verbose=0)
        idx = feat_df.index[self.lookback:]
        # output DataFrame aligned to forecast dates
        return pd.DataFrame(preds, index=idx, columns=self.asset_names)


# === BACKTESTER ===
class Backtester:
    """
    Executes an event-driven backtest with realistic transaction costs:
      - fixed half-spread slippage
      - square-root market impact based on ADV
      - weekly rebalancing on Mondays
      - inverse-volatility position sizing
    """
    def __init__(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame = None,
        spread_pct: float = 0.001,
        impact_coeff: float = 1e-3,
        init_cash: float = 1e6,
        adv_window: int = 20
    ):
        # sorted price series
        self.prices = prices.sort_index()
        self.volumes = volumes
        # transaction cost parameters
        self.spread_pct = spread_pct
        self.impact_coeff = impact_coeff
        # track cash as float
        self.cash = float(init_cash)
        # series to update cash over time
        self.cash_series = pd.Series(self.cash, index=self.prices.index)
        # DataFrame of share positions, floats to allow partial shares
        self.positions = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        # compute ADV for sqrt-impact model
        self.adv = volumes.rolling(window=adv_window).mean() if volumes is not None else None

    def execute_order(self, asset: str, date: pd.Timestamp, dollar_size: float):
        # mid-price at execution date
        price = self.prices.at[date, asset]
        # convert notional to share size
        size = dollar_size / price
        # slippage cost = half-spread * price
        slip = 0.5 * price * self.spread_pct * np.sign(size)
        # market impact ~ price * sqrt(size / ADV) if ADV available
        if self.adv is not None:
            adv_val = max(self.adv.at[date, asset], 1)
            impact = self.impact_coeff * price * np.sqrt(abs(size) / adv_val)
        else:
            impact = self.impact_coeff * price * abs(size)
        # fill price excludes impact, impact added separately
        fill_price = price + slip
        # total cost includes slippage and impact
        cost = fill_price * size + impact
        # decrement cash and propagate to future
        self.cash -= cost
        self.cash_series.loc[date:] -= cost
        # add shares from date onwards
        self.positions.loc[date:, asset] += size

    def run_backtest(self, signals: pd.DataFrame, vol: pd.DataFrame, threshold: float = 0.01):
        # loop over signal dates
        for date, sig_series in signals.iterrows():
            # only trade Mondays for weekly rebalance
            if date.weekday() != 0:
                continue
            # inverse-volatility weights
            inv_vol = 1 / vol.loc[date]
            inv_vol /= inv_vol.sum()
            # trade each asset exceeding threshold
            for asset, sig_val in sig_series.items():
                if abs(sig_val) < threshold:
                    continue
                # allocate cash proportionally and directionally
                alloc = self.cash_series.at[date] * inv_vol[asset] * np.sign(sig_val)
                self.execute_order(asset, date, alloc)
        return self.performance()

    def performance(self):
        # mark-to-market portfolio value
        port_val = (self.positions * self.prices).sum(axis=1) + self.cash_series
        # compute returns series
        returns = port_val.pct_change().fillna(0)
        # summary stats: total return and annualized Sharpe
        stats = {
            'Total Return': port_val.iloc[-1] / port_val.iloc[0] - 1,
            'Sharpe': returns.mean() / returns.std() * np.sqrt(252)
        }
        perf_df = pd.DataFrame({'PortfolioValue': port_val, 'Returns': returns})
        return perf_df, stats


if __name__ == '__main__':
    import yfinance as yf
    import matplotlib.pyplot as plt

    # 1) Data download for a basket of tech tickers
    tickers = ['AAPL','AMZN','MSFT','GOOG','NVDA','TSLA','META','AMD','INTC']
    raw = yf.download(tickers, start='2020-01-01', end='2025-05-01')
    price = raw['Close']
    volume = raw['Volume']

    # 2) Initialize handlers and engineer features
    dh = DataHandler(price, volume)              # load and preprocess data
    fe = FeatureEngineer(dh)                      # feature factory
    pca = fe.compute_pca_signals(3)              # top-3 principal components
    pairs = fe.find_cointegrated_pairs()         # identify mean-reverting pairs
    mom = fe.compute_momentum(5)                 # 5-day momentum signals
    vol_est = fe.compute_volatility(20)          # 20-day rolling volatility
    spread_z = fe.compute_spread_zscores(pairs[:3], 20)  # z-scores for top pairs
    # join signals on common dates
    signals = pd.concat([pca, mom, spread_z], axis=1).dropna()

    # 3) Train LSTM alpha model to forecast next-day returns
    target = dh.returns.loc[signals.index]       # true next-day returns
    alpha_model = LSTMAlphaModel(
        lookback=20,
        features=signals.columns.tolist(),
        asset_names=target.columns.tolist()
    )
    alpha_model.train(signals, target, epochs=10)    # fit model
    alpha_signals = alpha_model.predict(signals)    # generate forecasts

    # 4) Backtest strategy with event-driven engine
    bt = Backtester(price.loc[signals.index], volume.loc[signals.index])
    strat_df, strat_stats = bt.run_backtest(alpha_signals, vol_est, threshold=0.01)

    # 5) Compare to equal-weight benchmark
    init_cash = bt.cash_series.iloc[0]
    alloc = init_cash / len(tickers)
    shares = alloc / price.iloc[0]
    eq_val = (shares * price).sum(axis=1)
    eq_stats = {
        'Total Return': eq_val.iloc[-1]/eq_val.iloc[0] - 1,
        'Sharpe': eq_val.pct_change().mean()/eq_val.pct_change().std() * np.sqrt(252)
    }

    # 6) Plot performance
    plt.figure(figsize=(10,6))
    strat_df['PortfolioValue'].plot(label='Enhanced Stat-Arb')
    eq_val.plot(label='Equal-Weight')
    plt.title('Strategy vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.tight_layout()
    plt.show()
