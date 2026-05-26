from sklearn.linear_model import LassoLars
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import clone, linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb


from src.models.importance import linear_importance, permutation_importance, tree_importance
from src.models.hyperparameters import linreg_grid, gradient_boosting_grid, adaboost_grid, knn_grid, mlp_grid


class ModelInfo:
    def __init__(self, name, model, importance_fn, param_grid):
        self.name = name
        self.model = model
        self.importance_fn = importance_fn
        self.param_grid = param_grid

    def json(self):
        return {
            "name": self.name,
            "model": type(self.model).__name__,
            "param_grid": self.param_grid.__name__ if callable(self.param_grid) else str(self.param_grid)
        }
    
    def __repr__(self) -> str:
        json_repr = self.json()
        return f"ModelInfo(name={json_repr['name']}, model={json_repr['model']}, param_grid={json_repr['param_grid']})"

    # def explain(self, X_columns, **kwargs):
    #     if kwargs:
    #         return self.importance_fn(self.model, X_columns, **kwargs)
    #     return self.importance_fn(self.model, X_columns)
        # usage:
        # print(f"Importance top-1: {importance.index[0]} ({importance.iloc[0]})")
        # print(f"Importance top-2: {importance.index[1]} ({importance.iloc[1]})")
        # print(f"Importance top-3: {importance.index[2]} ({importance.iloc[2]})")


linreg = linear_model.Ridge(alpha=0.5)
gb = GradientBoostingRegressor(random_state=0, n_estimators=7819, min_samples_split=5, min_samples_leaf=8,
                               learning_rate=0.00285, alpha=0.24743)  # hp taken
ada_reg = AdaBoostRegressor(random_state=0, n_estimators=9289, learning_rate=0.5)  # hp taken
knn = KNeighborsRegressor(n_neighbors=4, p=1)  # hp taken
mlp = MLPRegressor(random_state=42, hidden_layer_sizes=(32, 16, 16, 8), max_iter=1000)


regressors = [
    (
        linreg,
        linear_importance,
        linreg_grid
    ),
    (
        gb,
        tree_importance,
        gradient_boosting_grid
    ),
    (
        ada_reg,
        tree_importance,
        adaboost_grid
    ),
    (
        knn,
        permutation_importance,
        knn_grid
    ),
    (
        mlp,
        permutation_importance,
        mlp_grid
    )
]

regressors = [(f"{type(model).__name__}", model, func, param_grid) for model, func, param_grid in regressors]
regressors = [ModelInfo(name, model, importance_fn, param_grid) for name, model, importance_fn, param_grid in regressors]
 