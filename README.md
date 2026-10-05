# Credit Card Fraud Detection using Machine Learning
Group 08 · Techno Main Salt Lake · B.Tech CSE (7th Sem)

An interactive ML system: enter a transaction (or upload a CSV) → validation → preprocessing →
trained model → **FRAUDULENT / LEGITIMATE** + fraud probability + risk level (Low / Medium / High).

## Quick start
```bash
pip install -r requirements.txt
# put creditcard.csv (Kaggle ULB dataset) into data/
python -m src.train          # trains 4 models, picks the best, saves model + metrics + figures (~1 min)
python -m pytest tests -q    # 19 test cases
streamlit run app.py         # launch the UI
```
A trained model, metrics and figures are already included, so `streamlit run app.py` works immediately.

## Project structure
```
app.py                  Streamlit UI (UI-1, UI-2, RES-1, RES-2, ML-3, ERR-1, VIS-1)
src/config.py           paths, feature list, thresholds
src/data_loader.py      DS-1  load / summarise / stratified split
src/preprocessing.py    PRE-1 cleaning + Amount scaling (fit on train only)
src/validation.py       UI-2, ERR-1 input validation
src/train.py            ML-1  training + model selection
src/predict.py          ML-2, ML-3 prediction service
src/risk.py             RES-2 risk bands
src/evaluate.py         ML-4, VIS-1 metrics and plots
tests/test_system.py    test cases named <MODULE>-TC<n>
models/ reports/ sample_data/   generated artefacts
```

## Results (held-out 20% test set, 55,133 transactions)
Selected model: **Random Forest** (chosen on a separate validation split, not on the test set).

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.065 | 0.905 | 0.122 | 0.9825 | 0.6919 |
| Decision Tree | 0.103 | 0.811 | 0.183 | 0.9149 | 0.4710 |
| Random Forest | 0.855 | 0.747 | 0.798 | 0.9679 | 0.7917 |
| Gradient Boosting | 0.386 | 0.800 | 0.521 | 0.9688 | 0.7347 |

Confusion matrix (Random Forest): TN=55,026 FP=12 FN=24 TP=71  ·  accuracy 99.9347%.
Only 0.17% of transactions are fraud, so accuracy alone is misleading - judge by precision/recall/F1/PR-AUC.

## Changes vs. the reference notebook (and why)
- Amount scaler is fitted on the **training split only** (notebook scaled before splitting → data leakage).
- **Stratified** split (fraud is rare, the notebook's random split could skew it).
- ROC-AUC computed from **probabilities**, not hard 0/1 labels.
- Used class weighting instead of discarding 50% of legitimate rows.
- Genetic-algorithm selection and XGBoost dropped to keep dependencies to the PRD's stack
  (scikit-learn); HistGradientBoosting is used as the boosting model.
- Added what the PRD requires but the notebook lacked: input validation, probability, risk level,
  saved model (joblib), Streamlit UI, batch CSV scoring, tests.

## Notes / limitations
- V1-V28 are anonymised PCA features, so real card details cannot be typed in; use the sample buttons or CSV upload.
- `Time` is dropped (seconds since the first transaction in the dataset - not meaningful for new input).
- Academic prototype: it does not authorise or block real transactions.
