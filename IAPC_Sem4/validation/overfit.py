# validation/overfit.py

from validation.metrics import compute_metrics

def compute_overfit_gap(model, X_train, y_train, X_test, y_test, model_name=""):
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    print(f"\n{'='*40}")
    print(f"Overfitting Analysis: {model_name}")
    train_m = compute_metrics(y_train, train_preds, label="TRAIN")
    test_m  = compute_metrics(y_test,  test_preds,  label="TEST")

    gap = {
        'MAE_gap':  round(test_m['MAE']  - train_m['MAE'],  6),
        'RMSE_gap': round(test_m['RMSE'] - train_m['RMSE'], 6),
        'R2_gap':   round(train_m['R2']  - test_m['R2'],    4),
        'DA_gap':   round(train_m['DirectionalAccuracy'] - 
                          test_m['DirectionalAccuracy'], 4)
    }

    print(f"\n  [GAP] R2_gap={gap['R2_gap']}  DA_gap={gap['DA_gap']}")
    if gap['R2_gap'] > 0.3:
        print(f"  ⚠ WARNING: Likely overfitting detected.")
    else:
        print(f"  ✓ Generalization looks acceptable.")

    return {'train': train_m, 'test': test_m, 'gap': gap}

def run_overfit_analysis(trained_models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in trained_models.items():
        results[name] = compute_overfit_gap(
            model, X_train, y_train, X_test, y_test, model_name=name)
    return results