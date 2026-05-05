
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

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
        highest_p = 0
        if X_test is not None and len(X_test) >= 8:
            n_test = len(X_test)
            q1_idx = n_test // 4
            q3_idx = 3 * n_test // 4
            q2_idx = 2 * n_test // 4
            if q3_idx > q1_idx:
                q_slope = (y_test[q3_idx] - y_test[q1_idx]) / (X_test[q3_idx] - X_test[q1_idx] + 1e-10)
            highest_p = np.argmax(pred) - q2_idx
        return [mape, rmse, q_slope, highest_p]
    
    def calc_trend(coef, y_mean, n_points):
        return coef * n_points / (y_mean + 1e-10) * 100
    
    # 1. Regresja liniowa
    try:
        m = LinearRegression().fit(X_train.reshape(-1, 1), y_train_orig)
        pred = m.predict(X_test.reshape(-1, 1))
        score = r2_score(y_test_orig, pred)
        trend = calc_trend(m.coef_[0] if m.coef_.ndim > 0 else m.coef_, y_train_orig.mean(), train_size)
        err = get_err(y_test_orig, pred, X_test)
        curvature = 0.0
        deriv = float(m.coef_[0] if m.coef_.ndim > 0 else m.coef_)
        
        if score > best_score:
            best_score = score
            best = "Linear"
            best_feat = [trend] + err + [curvature]
            derivative = deriv
    except Exception:
        pass
    
    # 2. Regresja wykładnicza
    if best_score < 0.99:
        try:
            y_safe = np.maximum(y_train_orig, 1e-10)
            y_log = np.log(y_safe)
            if np.isnan(y_log).any():
                raise ValueError("NaN in log")
            m = LinearRegression().fit(X_train.reshape(-1, 1), y_log)
            b = m.coef_[0] if m.coef_.ndim > 0 else m.coef_
            a = np.exp(m.intercept_)
            pred = a * np.exp(b * X_test)
            score = r2_score(y_test_orig, pred)
            
            if score > best_score and score > 0.3:
                best_score = score
                best = "Exponential"
                trend = b * 100
                err = get_err(y_test_orig, pred, X_test)
                curvature = b * b
                deriv = float(b * a * np.exp(b * X_train.mean()))
                best_feat = [trend] + err + [curvature]
                derivative = deriv
        except Exception:
            pass
    
    # 3. Regresja wielomianowa (stopień 2)
    if best_score < 0.99:
        try:
            poly = PolynomialFeatures(degree=2)
            X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
            X_test_poly = poly.transform(X_test.reshape(-1, 1))
            
            m = LinearRegression().fit(X_train_poly, y_train_orig)
            pred = m.predict(X_test_poly)
            score = r2_score(y_test_orig, pred)
            
            if score > best_score:
                best_score = score
                best = "Polynomial"
                coefs = m.coef_.flatten() if m.coef_.ndim > 0 else np.array([float(m.coef_)])
                coef_linear = coefs[1] if len(coefs) > 1 else 0.0
                coef_quadratic = coefs[2] if len(coefs) > 2 else 0.0
                
                trend = calc_trend(coef_linear, y_train_orig.mean(), train_size)
                err = get_err(y_test_orig, pred, X_test)
                curvature = coef_quadratic
                deriv = float(coef_linear + 2 * coef_quadratic * X_train.mean())
                best_feat = [trend] + err + [curvature]
                derivative = deriv
        except Exception:
            pass
    
    return best, best_feat, derivative, best_score
