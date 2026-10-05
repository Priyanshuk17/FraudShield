"""ML-1: Model training pipeline.  Run:  python -m src.train"""
import json
import time
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .data_loader import load_dataset, dataset_summary, split_data
from .preprocessing import clean, fit_scaler, apply_scaler, save_scaler
from . import evaluate as ev


def candidate_models():
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                                class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=10, max_samples=0.3,
                                                min_samples_leaf=3, n_jobs=-1,
                                                class_weight="balanced_subsample",
                                                random_state=rs),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1,
                                                            max_iter=200, class_weight="balanced",
                                                            random_state=rs),
    }


def main():
    t0 = time.time()
    C.MODEL_DIR.mkdir(exist_ok=True); C.REPORT_DIR.mkdir(exist_ok=True)

    print("[1/6] Loading dataset ...")
    raw = load_dataset()
    summary = dataset_summary(raw)
    print("     ", summary)
    ev.plot_class_distribution(raw[C.TARGET])
    ev.plot_amount_distribution(raw)
    ev.plot_correlation(raw)

    print("[2/6] Cleaning + stratified split ...")
    df = clean(raw)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      train={len(X_train):,} test={len(X_test):,} "
          f"fraud(train)={int(y_train.sum())} fraud(test)={int(y_test.sum())}")

    print("[3/6] Scaling Amount (fit on train only) ...")
    scaler = fit_scaler(X_train)
    Xtr, Xte = apply_scaler(X_train, scaler), apply_scaler(X_test, scaler)

    print("[4/6] Model selection on a validation split ...")
    Xa, Xv, ya, yv = train_test_split(Xtr, y_train, test_size=0.2, stratify=y_train,
                                      random_state=C.RANDOM_STATE)
    models, val_ap, test_results, test_probas = candidate_models(), {}, {}, {}
    for name, m in models.items():
        s = time.time()
        fitted = clone(m).fit(Xa, ya)
        val_ap[name] = float(average_precision_score(yv, fitted.predict_proba(Xv)[:, 1]))
        print(f"      {name:20s} val PR-AUC={val_ap[name]:.4f} ({time.time()-s:.0f}s)")
    best_name = max(val_ap, key=val_ap.get)
    print(f"      -> selected: {best_name}")

    print("[5/6] Final fit of all models on full training set + test evaluation ...")
    final = {}
    for name, m in models.items():
        final[name] = clone(m).fit(Xtr, y_train)
        proba = final[name].predict_proba(Xte)[:, 1]
        test_probas[name] = proba
        test_results[name] = ev.compute_metrics(y_test, proba)
        r = test_results[name]
        print(f"      {name:20s} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} ROC-AUC={r['roc_auc']:.4f} PR-AUC={r['pr_auc']:.4f}")

    print("[6/6] Saving model, metrics and figures ...")
    best = final[best_name]
    joblib.dump(best, C.MODEL_PATH)
    save_scaler(scaler)
    ev.plot_confusion(test_results[best_name]["confusion_matrix"], best_name)
    ev.plot_roc(y_test, test_probas); ev.plot_pr(y_test, test_probas)
    ev.plot_model_comparison(test_results)
    ev.plot_feature_importance(best, best_name)

    # sample transactions for the UI demo (taken from the held-out test set)
    C.SAMPLE_DIR.mkdir(exist_ok=True)
    te = X_test.copy(); te[C.TARGET] = y_test.values
    rng = np.random.RandomState(C.RANDOM_STATE)
    samples = [te[te[C.TARGET] == 0].sample(10, random_state=rng),
               te[te[C.TARGET] == 1].sample(10, random_state=rng)]
    import pandas as pd
    pd.concat(samples).to_csv(C.SAMPLE_DIR / "sample_transactions.csv", index=False)

    meta = {"selected_model": best_name, "features": C.FEATURES,
            "decision_threshold": C.DECISION_THRESHOLD,
            "trained_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "validation_pr_auc": val_ap}
    C.METADATA_PATH.write_text(json.dumps(meta, indent=2))
    C.METRICS_PATH.write_text(json.dumps(
        {"dataset": summary, "selected_model": best_name, "test_results": test_results},
        indent=2))
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
