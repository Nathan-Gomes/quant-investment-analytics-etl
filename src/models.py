import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["return_1", "return_5", "return_20", "volatility_20", "market_return_20", "volume_ratio"]


def feature_frame(group, benchmark, volume, horizon):
    r = group.set_index("date").net_return.sort_index()
    frame = pd.DataFrame({"return_1": r, "return_5": (1 + r).rolling(5).apply(np.prod, raw=True) - 1,
                          "return_20": (1 + r).rolling(20).apply(np.prod, raw=True) - 1,
                          "volatility_20": r.rolling(20).std() * np.sqrt(252),
                          "market_return_20": (1 + benchmark).rolling(20).apply(np.prod, raw=True) - 1,
                          "volume_ratio": volume / volume.rolling(20).mean()})
    # The label at t contains returns t+1 through t+h; purge h rows at every split.
    frame["target"] = r.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    frame["target_end"] = pd.Series(r.index, index=r.index).shift(-horizon)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def fit_models(daily, prices, config):
    benchmark = daily[daily.portfolio_id == "Benchmark"].set_index("date").net_return
    volume = prices[prices.ticker == config["benchmark"]].set_index("date").volume
    horizon = config["forecast_days"]
    scores, predictions, coefficients, audits = [], [], [], []
    candidates = {"Linear regression": (LinearRegression(), {}),
                  "Ridge": (Ridge(), {"model__alpha": [0.1, 1., 10., 100.]}),
                  "Lasso": (Lasso(max_iter=20000), {"model__alpha": [0.00001, 0.0001, 0.001]}),
                  "ElasticNet": (ElasticNet(max_iter=20000),
                                 {"model__alpha": [0.0001, 0.001], "model__l1_ratio": [0.25, 0.75]})}
    for name, group in daily.groupby("portfolio_id"):
        if name == "Benchmark":
            continue
        frame = feature_frame(group, benchmark, volume, horizon)
        split = int(len(frame) * (1 - config["test_fraction"]))
        train, test = frame.iloc[:split - horizon], frame.iloc[split:]
        if len(train) < 150 or len(test) < 30:
            raise ValueError("Insufficient history for purged chronological validation")
        if train.target_end.max() >= test.index.min():
            raise ValueError("Training labels overlap the test period")
        cv = TimeSeriesSplit(n_splits=4, gap=horizon)
        for fold, (tr, va) in enumerate(cv.split(train)):
            if train.iloc[tr].target_end.max() >= train.iloc[va].index.min():
                raise ValueError("Cross-validation label leakage")
            audits.append({"portfolio_id": name, "fold": fold, "train_start": train.iloc[tr].index.min(),
                           "last_train_label": train.iloc[tr].target_end.max(),
                           "validation_start": train.iloc[va].index.min(), "validation_end": train.iloc[va].index.max()})
        models = {}
        for model_name, (model, grid) in candidates.items():
            search = GridSearchCV(Pipeline([("scale", StandardScaler()), ("model", model)]), grid,
                                  cv=cv, scoring="neg_mean_squared_error", n_jobs=1)
            search.fit(train[FEATURES], train.target)
            models[model_name] = (np.maximum(search.predict(test[FEATURES]), 0), -search.best_score_, str(search.best_params_))
            for feature, value in zip(FEATURES, search.best_estimator_.named_steps["model"].coef_):
                coefficients.append({"portfolio_id": name, "model": model_name,
                                     "feature": feature, "coefficient": value})
        selected = min(models, key=lambda key: models[key][1])
        models["Persistence baseline"] = (test.volatility_20.to_numpy(), np.nan, "Trailing 20-day volatility")
        for model_name, (prediction, cv_mse, params) in models.items():
            scores.append({"portfolio_id": name, "model": model_name,
                           "rmse": np.sqrt(mean_squared_error(test.target, prediction)),
                           "r2": r2_score(test.target, prediction), "cv_mse": cv_mse,
                           "selected_by_cv": model_name == selected, "parameters": params,
                           "train_end": train.index[-1], "last_train_label": train.target_end.max(),
                           "test_start": test.index[0], "test_end": test.index[-1], "test_rows": len(test)})
            predictions.extend({"date": date, "portfolio_id": name, "model": model_name,
                                "actual": actual, "prediction": value}
                               for date, actual, value in zip(test.index, test.target, prediction))
    return pd.DataFrame(scores), pd.DataFrame(predictions), pd.DataFrame(coefficients), pd.DataFrame(audits)
