import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

N_PERIODS = 5  # liczba okresów do podziału szeregu czasowego


def predict_regression(X, y_train, reg_type):
    """Predict regression values for given type. Returns y_pred array."""
    X = X.flatten()
    n = len(X)
    X_reshape = X.reshape(-1, 1)
    
    if reg_type == "Linear":
        model = LinearRegression().fit(X_reshape[:int(n*0.8)], y_train[:int(n*0.8)])
        y_pred = model.predict(X_reshape)
    elif reg_type == "Exponential":
        y_safe = np.maximum(y_train, 1e-10)
        y_log = np.log(y_safe)
        model = LinearRegression().fit(X_reshape[:int(n*0.8)], y_log[:int(n*0.8)])
        b = model.coef_[0] if model.coef_.ndim > 0 else model.coef_
        a = np.exp(model.intercept_)
        y_pred = a * np.exp(b * X)
    elif reg_type == "Polynomial":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_reshape)
        model = LinearRegression().fit(X_poly[:int(n*0.8)], y_train[:int(n*0.8)])
        y_pred = model.predict(poly.transform(X_reshape))
    else:
        y_pred = y_train
    
    return y_pred


def best_regression(X, y):
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64).flatten()
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return None, None, None, None

    n = len(y)
    train_size = int(n * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train_orig, y_test_orig = y[:train_size], y[train_size:]

    best = None
    best_score = -np.inf
    best_feat = None
    derivative = None

    def get_err(y_test, pred, X_test=None):
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = np.mean(np.abs(y_test - pred) / (np.abs(y_test) + 1e-10)) * 100
        q_slope = 0.0
        if X_test is not None and len(X_test) >= 8:
            n_test = len(X_test)
            q1_idx = n_test // 4
            q3_idx = 3 * n_test // 4
            if q3_idx > q1_idx:
                q_slope = (y_test[q3_idx] - y_test[q1_idx]) / (X_test[q3_idx] - X_test[q1_idx] + 1e-10)
        return [mape, rmse, q_slope]

    def calc_trend(coef, y_mean, n_points):
        return coef * n_points / (y_mean + 1e-10) * 100

    def compute_period_rmse(y_full, pred_full, n_periods=N_PERIODS):
        """Split time series into n_periods and compute RMSE for each period."""
        chunk_size = len(y_full) // n_periods
        rmses = []
        for i in range(n_periods):
            start = i * chunk_size
            if i == n_periods - 1:
                end = len(y_full)
            else:
                end = start + chunk_size
            y_chunk = y_full[start:end]
            pred_chunk = pred_full[start:end]
            rmse = np.sqrt(mean_squared_error(y_chunk, pred_chunk))
            rmses.append(rmse)
        return rmses

    def compute_period_highest_p(pred_full, n_periods=N_PERIODS):
        """Split time series into n_periods and compute highest_p for each period."""
        chunk_size = len(pred_full) // n_periods
        highest_ps = []
        for i in range(n_periods):
            start = i * chunk_size
            if i == n_periods - 1:
                end = len(pred_full)
            else:
                end = start + chunk_size
            pred_chunk = pred_full[start:end]
            mid_idx = len(pred_chunk) // 2
            hp = np.argmax(pred_chunk) - mid_idx
            highest_ps.append(float(hp))
        return highest_ps

    # 1. Regresja liniowa
    try:
        pred_full = predict_regression(X, y_train_orig, "Linear")
        pred_test = pred_full[train_size:]
        score = r2_score(y_test_orig, pred_test)
        trend = calc_trend(LinearRegression().fit(X_train.reshape(-1, 1), y_train_orig).coef_[0], y_train_orig.mean(), train_size)
        err = get_err(y_test_orig, pred_test, X_test)
        curvature = 0.0
        deriv = float(LinearRegression().fit(X_train.reshape(-1, 1), y_train_orig).coef_[0])
        period_rmses = compute_period_rmse(y, pred_full)
        period_highest_ps = compute_period_highest_p(pred_full)

        if score > best_score:
            best_score = score
            best = "Linear"
            best_feat = [trend, err[0], err[2], curvature] + period_highest_ps + period_rmses
            derivative = deriv
    except Exception:
        pass

    # 2. Regresja wykładnicza
    if best_score < 0.99:
        try:
            pred_full = predict_regression(X, y_train_orig, "Exponential")
            pred_test = pred_full[train_size:]
            score = r2_score(y_test_orig, pred_test)

            if score > best_score and score > 0.3:
                best_score = score
                best = "Exponential"
                b = np.log(y_train_orig[1]/y_train_orig[0]) if len(y_train_orig) > 1 else 0.0
                trend = b * 100
                err = get_err(y_test_orig, pred_test, X_test)
                curvature = b * b
                deriv = float(b * np.exp(b * X_train.mean()))
                period_rmses = compute_period_rmse(y, pred_full)
                period_highest_ps = compute_period_highest_p(pred_full)
                best_feat = [trend, err[0], err[2], curvature] + period_highest_ps + period_rmses
                derivative = deriv
        except Exception:
            pass

    # 3. Regresja wielomianowa (stopień 2)
    if best_score < 0.99:
        try:
            pred_full = predict_regression(X, y_train_orig, "Polynomial")
            pred_test = pred_full[train_size:]
            score = r2_score(y_test_orig, pred_test)

            if score > best_score:
                best_score = score
                best = "Polynomial"
                m = LinearRegression().fit(PolynomialFeatures(degree=2).fit_transform(X_train.reshape(-1, 1)), y_train_orig)
                coefs = m.coef_.flatten() if m.coef_.ndim > 0 else np.array([float(m.coef_)])
                coef_linear = coefs[1] if len(coefs) > 1 else 0.0
                coef_quadratic = coefs[2] if len(coefs) > 2 else 0.0

                trend = calc_trend(coef_linear, y_train_orig.mean(), train_size)
                err = get_err(y_test_orig, pred_test, X_test)
                curvature = coef_quadratic
                deriv = float(coef_linear + 2 * coef_quadratic * X_train.mean())
                period_rmses = compute_period_rmse(y, pred_full)
                period_highest_ps = compute_period_highest_p(pred_full)
                best_feat = [trend, err[0], err[2], curvature] + period_highest_ps + period_rmses
                derivative = deriv
        except Exception:
            pass

    return best, best_feat, derivative, best_score
