import os

# ─── Turn off oneDNN and enable deterministic ops ───
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISM'] = '1'
os.environ['PYTHONHASHSEED'] = '0'


import random
import numpy as np
import pandas as pd

# === Statistical & ML imports ===
import tensorflow as tf
from sklearn.decomposition import PCA  # extract orthogonal factors from returns
from statsmodels.tsa.stattools import coint  # cointegration test for mean-reversion signals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, Zeros
from tensorflow.keras.callbacks import EarlyStopping

# ─── Global seed ───
SEED = 70
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Pairwise ranking loss
def pairwise_ranking_loss(y_true, y_pred):
    diffs = tf.expand_dims(y_pred, -1) - tf.expand_dims(y_pred, -2)
    true_diffs = tf.expand_dims(y_true, -1) - tf.expand_dims(y_true, -2)
    signs = tf.sign(true_diffs)
    loss = tf.nn.relu(1.0 - signs * diffs)
    return tf.reduce_mean(loss)



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
        pca = PCA(n_components=n_components,
                  svd_solver='full',  # deterministic solver
                  random_state=SEED)  # if you ever use randomized SVD
        vals = pca.fit_transform(self.data.returns.values)
        cols = [f"PC{i + 1}" for i in range(n_components)]
        # align PC time series with original dates
        return pd.DataFrame(vals, index=self.data.returns.index, columns=cols)

    def find_cointegrated_pairs(self, pvalue_threshold=0.2) -> list:
        # detect pairs where price series share a long-run equilibrium
        assets = self.data.prices.columns
        pairs = []
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
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

    def compute_technical_indicators(self) -> pd.DataFrame:
        # Compute simple moving averages and returns
        sma_10 = self.data.prices.rolling(window=10).mean()
        sma_50 = self.data.prices.rolling(window=50).mean()
        rsi = 100 - 100 / (1 + self.data.returns.rolling(14).mean() / self.data.returns.rolling(14).std())
        momentum_10 = self.data.prices.pct_change(periods=10).shift(1)
        sma_diff = sma_10 - sma_50
        return pd.concat([sma_diff, rsi, momentum_10], axis=1, keys=['sma_diff', 'rsi', 'momentum'])

    def compute_macro_features(self, tickers_df: pd.DataFrame) -> pd.DataFrame:
        spy = tickers_df['SPY']
        vix = tickers_df['VIXY']
        spy_ret = spy.pct_change().rename('SPY_ret')
        vix_ret = vix.pct_change().rename('VIX_ret')

        sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLU', 'IWM']
        sector_ret = tickers_df[sector_etfs].pct_change()
        dispersion = sector_ret.std(axis=1).rename('sector_dispersion')

        return pd.concat([spy_ret, vix_ret, dispersion], axis=1)


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
            X.append(data_arr[idx - self.lookback:idx])
            # next-day target vector
            Y.append(tgt_arr[idx])
        return np.array(X), np.array(Y)

    def build_model(self):
        # explicit Input layer to satisfy Keras best practices
        self.model = Sequential([
            Input(shape=(self.lookback, len(self.features))),
            LSTM(64,
                 name="lstm_layer",  # capture temporal dependencies
                 kernel_initializer=GlorotUniform(seed=SEED),
                 recurrent_initializer=GlorotUniform(seed=SEED),
                 bias_initializer=Zeros()),
            Dense(self.n_assets,
                  name="output_layer",  # one output per asset
                  kernel_initializer=GlorotUniform(seed=SEED),
                  bias_initializer=Zeros())
        ])
        # Adam optimizer balances speed and stability
        self.model.compile(optimizer=Adam(1e-3), loss=pairwise_ranking_loss)

    def train(self, feat_df: pd.DataFrame, target_df: pd.DataFrame,
              epochs=100, batch_size=32):
        # prepare inputs and outputs
        X, Y = self.prepare_sequences(feat_df, target_df)
        # Normalize features
        X_mean = X.mean(axis=(0, 1), keepdims=True)
        X_std = X.std(axis=(0, 1), keepdims=True)
        X = (X - X_mean) / X_std
        self.build_model()
        # verbose=2 prints one line per epoch
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(
            X, Y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[early_stopping],
            verbose=0
        )

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
                raw_alloc = self.cash_series.at[date] * inv_vol[asset] * sig_val
                alloc = np.clip(raw_alloc, -0.05 * self.cash_series.at[date], 0.05 * self.cash_series.at[date])
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
    import pandas as pd

    # 1) Data download for a basket of tech tickers
    tickers = ['AAPL', 'AMZN', 'MSFT', 'GOOG', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC', 'NFLX', 'ADBE', 'SPY', 'VIXY',
               'XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLU', 'IWM']
    raw = yf.download(tickers, start='2023-01-01', end='2024-12-31')
    price = raw['Close']
    volume = raw['Volume']

    # 2) Initialize handlers and engineer features
    dh = DataHandler(price, volume)  # load and preprocess data
    fe = FeatureEngineer(dh)  # feature factory
    pca = fe.compute_pca_signals(5)  # top-3 principal components
    pairs = fe.find_cointegrated_pairs()  # identify mean-reverting pairs
    mom = fe.compute_momentum(5)  # 5-day momentum signals
    vol_est = fe.compute_volatility(20)  # 20-day rolling volatility
    spread_z = fe.compute_spread_zscores(pairs[:3], 20)  # z-scores for top pairs
    tech = fe.compute_technical_indicators()
    signals = pd.concat([pca, mom, spread_z, tech], axis=1).dropna()
    target = dh.returns.loc[signals.index].rank(axis=1, pct=True) * 2 - 1  # true next-day returns
    vol_est = vol_est.loc[signals.index]

    # ---- Split data into train and test by date ----
    split_date = pd.Timestamp('2024-01-01')
    train_signals = signals.loc[:split_date]
    train_target = target.loc[:split_date]
    test_signals = signals.loc[split_date:]
    test_vol = vol_est.loc[split_date:]
    test_prices = price.loc[test_signals.index]

    # 3) Train LSTM alpha model to forecast next-day returns
    alpha_model = LSTMAlphaModel(
        lookback=20,
        features=signals.columns.tolist(),
        asset_names=target.columns.tolist()
    )
    alpha_model.train(train_signals, train_target, epochs=100)  # fit model
    alpha_signals_test = alpha_model.predict(test_signals)  # generate forecasts

    # Smooth predictions before backtest
    alpha_signals_test = alpha_signals_test.ewm(span=3).mean()

    # 4) Backtest strategy with event-driven engine
    bt = Backtester(price.loc[test_signals.index], volume.loc[test_signals.index])
    strat_df, strat_stats = bt.run_backtest(alpha_signals_test, test_vol, threshold=0.01)

    # 6) Equal‐Weight Benchmark on TEST data
    init_cash = bt.cash_series.iloc[0]
    first_prices = test_prices.iloc[0]
    per_asset_cash = init_cash / len(tickers)
    shares = per_asset_cash / first_prices
    eq_val = (shares * test_prices).sum(axis=1)
    eq_rets = eq_val.pct_change().fillna(0)
    eq_stats = {
        'Total Return': eq_val.iloc[-1] / eq_val.iloc[0] - 1,
        'Sharpe': eq_rets.mean() / eq_rets.std() * np.sqrt(252)
    }

    # ─── 7) Print Sharpe & Improvement ─────────────────────────────────────────
    print(f"Stat-Arb Sharpe:   {strat_stats['Sharpe']:.2f}")
    print(f"Equal-Wt Sharpe:   {eq_stats['Sharpe']:.2f}")
    print("Sharpe Improvement: {:.2f}".format(
        strat_stats['Sharpe'] - eq_stats['Sharpe']
    ))

    # ─── 8) Equity Curves ───────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    strat_df['PortfolioValue'].plot(label='Enhanced Stat-Arb')
    eq_val.plot(label='Equal-Weight')
    plt.title(f'Equity Curves (Test from {split_date.date()})')
    plt.xlabel('Date');
    plt.ylabel('Portfolio Value')
    plt.legend();
    plt.tight_layout();
    plt.show()

    # ─── 9) Bar Charts: Total Return & Sharpe ──────────────────────────────────
    perf_df = pd.DataFrame({
        'Enhanced Stat-Arb': [strat_stats['Total Return'], strat_stats['Sharpe']],
        'Equal-Weight': [eq_stats['Total Return'], eq_stats['Sharpe']]
    }, index=['Total Return', 'Sharpe'])
    perf_df.T.plot(kind='bar', subplots=True, layout=(1, 2), figsize=(12, 4), legend=False, title=['Total Return', 'Sharpe Ratio'])
    plt.tight_layout();
    plt.show()
