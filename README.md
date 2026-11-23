# Can forum and news posts predict a stock's direction?

Team project (6 members) for *Big Data and Business Analytics*, National Taiwan University, spring 2025.
Target stock: Innolux (TWSE: 3481).

## What it does
1. **Filter** — keep posts that mention the company (keywords such as 群創, 3481, INNOLUX, 面板雙虎): 9,139 posts from PTT and news.
2. **Label** — tag each post up / down / flat by the price move *N* trading days later at several thresholds (±1%, ±3%, ±5%; N = 1, 5, 10, 20).
3. **Features** — Chinese word segmentation, n-grams, chi-square keyword scoring and TF-IDF vectors.
4. **Models** — six classifiers (RF, NB, SVM, XGBoost, KNN, DT) and soft-voting ensembles.
5. **Rolling backtest** — each day, train on the previous three months of posts and predict the day-5 direction; only act when the predicted probabilities differ by at least the confidence threshold.
6. **LLM comparison** — zero-shot labels from TAIDE-LX-7B-Chat.

## Results
| Setting | Accuracy |
|---|---|
| Best single model (random forest), held-out test split | 69.17% |
| Ensemble RF + NB + XGBoost, held-out test split | **70.49%** |
| Same ensemble, day-by-day rolling backtest 2023-01 → 2025-02 (confidence 0.3; acted on 3,407 of 4,963 post-level predictions) | **59.99%** |

The backtest is the stricter number: it only uses information available before each prediction.

## Known limitations
These are course-project numbers and are likely optimistic:
- The 70% figures use a random train/test split. Posts from the same days share overlapping 5-day labels, so the split is not independent in time.
- Chi-square keyword selection was fitted on all data before the split.
- In the rolling backtest, the most recent training posts are labeled with prices that were not yet known on the prediction day, and the confidence threshold was chosen on the same period.
- There is no majority-class baseline in the table.

A time-ordered re-evaluation (selection inside the training window, an embargo on unresolved labels, and a majority baseline) is planned.

## My part
Preprocessing (`preprocess/`): keyword filtering, price-move labeling, n-gram / TF-IDF / chi-square features.

## Layout
```
preprocess/   filtering, labeling and feature scripts
modeling/     data_process.py (team data prep), stock_project.ipynb (models + backtest), llm_predict.py (TAIDE)
```

## Data and how the pieces connect
The course dataset (forum posts and stock prices provided by the instructor) is **not included** and may not be redistributed; notebook outputs were cleared for the same reason.

This is the code as the team used it, not a one-command pipeline:
- `preprocess/` scripts read the raw course CSVs and write their outputs to the working directory.
- `modeling/stock_project.ipynb` reads feature/label files (`5d_3_trend_y.csv`, `5d_3_trend_time.csv`, `5d_3_trend_source.csv`, …) that a teammate exported from the vectorization step; that export script is not in this repo.
- `modeling/llm_predict.py` expects a prepared `train_n20.jsonl` (posts with 20-day labels), also produced outside this repo.

Results above are from the team's runs and final presentation.
