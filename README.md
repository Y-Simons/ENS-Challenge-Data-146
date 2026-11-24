# ENS Challenge Data 146 — stock identification from market data sequences

Machine-learning project — NEOMA M.Sc Finance & Big Data, April 2024.

## Problem

[Challenge Data](https://challengedata.ens.fr/) (ENS), challenge 146: given an
anonymised sequence of order-book events (observation `obs_id`: venue, order
id, action, side, price, bid, ask, bid/ask sizes, trade flag, flux), predict
which of **24 stocks** (`eqt_code_cat`) it comes from.

The training set has 16.08 million events. Data must be downloaded from the
challenge website and is not included here.

## Approach

1. **Preprocessing**: categorical encoding of `side`, `action` and `trade`;
   `log1p` transform of price, bid, ask, sizes and flux; one sequence per
   observation, sorted by `order_id`, padded/truncated to 100 events.
2. **Model** (Keras): embeddings for venue, action and trade, concatenated with
   the 6 numerical features → 2 bidirectional GRU layers (64 units) → dense
   layer (64, SELU) → softmax over 24 classes. 121k parameters
   ([summary](results/model_summary.png)).
3. **Training**: Adam (lr 3e-3), batch size 1000, 20% validation split, early
   stopping (patience 5), checkpointing and learning-rate reduction on plateau.

## Results

- Validation accuracy plateaus at **about 56–58%** across 24 classes (chance
  level ≈ 4%); training accuracy reaches about 68%
  ([accuracy](results/accuracy_curves.png), [loss](results/loss_curves.png)).
- A later k-fold run ([console](results/console_kfold_run.png)) shows about
  70% validation accuracy on the second fold. That figure overstates
  performance: the model was not re-initialised between folds, so the second
  fold's validation data had already been seen during training on the first
  fold.
- Ideas noted for improvement: feature engineering (lags, PCA), external
  features and further tuning; training was limited by local compute.

## Repository

```
src/gru_stock_identification.py   final training script
notebooks/exploration_MLG.ipynb   data loading and exploratory analysis (with outputs)
results/                          model summary, training curves, histograms
```

To run: put the challenge CSV files in `data/`, then
`python src/gru_stock_identification.py` (TensorFlow 2.x, scikit-learn,
pandas, matplotlib).
