import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

class QEPredictor:
    def __init__(self, *, alpha, safety, gb_p, xgb_p, seed=42):
        self.alpha, self.safety = alpha, safety
        self.gb  = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, random_state=seed, **gb_p)
        base = dict(objective="reg:quantileerror",
                    quantile_alpha=alpha, n_jobs=1, random_state=seed)
        base.update(xgb_p)
        self.xgb = xgb.XGBRegressor(**base)
        self.cols = None

    def _enc(self, X, fit=False):
        Xd = pd.get_dummies(X, drop_first=True)
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]
        if fit:
            self.cols = Xd.columns
        else:
            miss = self.cols.difference(Xd.columns)
            for c in miss: Xd[c] = 0
            Xd = Xd[self.cols]
        return Xd.astype(float)

    def fit(self, X, y):
        self.gb.fit(self._enc(X, True), y)
        self.xgb.fit(self._enc(X), y, verbose=False)

    def predict(self, X):
        Xd = self._enc(X, False)
        p  = np.maximum(self.gb.predict(Xd), self.xgb.predict(Xd))
        return p * self.safety
