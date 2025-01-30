from catboost import CatBoostClassifier
import numpy as np
from types import SimpleNamespace
from typing import Dict, Any, List
from sklearn.model_selection import GridSearchCV

from src.evaluation import evaluator

def fit_and_optimize(
        labels: np.ndarray, 
        features: np.ndarray, 
        base_model: Any, 
        param_grid: Dict[str, List[Any]], 
        cv: int = 5, 
        scoring_fit: str='MultiLogloss') -> GridSearchCV:
    gs = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(labels, features)
    return fitted_model

def train_cat_boost_classifier(X_train, y_train, X_test, y_test, params: SimpleNamespace) -> Any:
    base_model = CatBoostClassifier(**vars(params.model.baseline))

    print(f"Looking for best combination of parameters with objective {params.model.tune.metric}. Parameters are {params.model.tune.search.to_dict().keys()}")
    search = fit_and_optimize(X_train, 
                              y_train, 
                              base_model=base_model,
                              param_grid=params.model.tune.search.to_dict(),
                              cv=params.model.tune.cv,
                              scoring_fit=params.model.tune.metric)

    print(f"Evaluating the performance of the model")
    evaluator.evaluate_search(search, plot_params_name=['learning_rate', 'depth'], to_mlflow = True)

    best_model = search.best_estimator_
    metrics = evaluator.evaluate_classifier(best_model, X_test, y_test)
    print(f"Logging the pipeline to MLflow")
    print(search.best_params_)
    print(metrics)

    return best_model