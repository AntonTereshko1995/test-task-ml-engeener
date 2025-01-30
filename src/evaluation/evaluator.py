
import sklearn_evaluation
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV

def evaluate_search(search: GridSearchCV, plot_params_name: List[str], subset_params_name: List[str] = None, to_mlflow: bool = False):
    if len(plot_params_name) != 2:
        ValueError("Exactly two parameters have to be indicated for the evaluation (they are displayed on a grid).")

    for param in plot_params_name:
        if param not in search.cv_results_['params'][0].keys():
            ValueError(f"Parameter {param} wasn't used in the parameters search.")

    if subset_params_name:
        # Filter dictionary by the provided params
        graph_by_params = dict(filter(lambda item: item[0] in subset_params_name and len(item[1]) > 1, search.param_grid.items()))
    else:
        # Filter dictionary by the parameters not used on each plot
        graph_by_params = dict(filter(lambda item: item[0] not in plot_params_name and len(item[1]) > 1, search.param_grid.items()))

    # Gettings parameters and its values
    graph_by_params_values = graph_by_params.values()
    graph_by_params_keys = list(graph_by_params.keys())
    graph_by_params_dim = len(graph_by_params_keys)

    if graph_by_params_dim == 0:
        fig = plt.figure()
        ax = fig.add_subplot()
        sklearn_evaluation.plot.grid_search(search.cv_results_, 
                                            change=(tuple(plot_params_name)), 
                                            ax=ax)
        ax.plot()
    else:
        for values in np.array(np.meshgrid(*graph_by_params_values)).T.reshape(-1, graph_by_params_dim):
            subset = { graph_by_params_keys[i]: [values[i]] for i in range(0, graph_by_params_dim) }
            title = ', '.join([f"{item[0]}={item[1][0]}" for item in subset.items()])

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title(title)
            sklearn_evaluation.plot.grid_search(search.cv_results_, 
                                                change=(tuple(plot_params_name)), 
                                                subset=subset, 
                                                ax=ax)
            if to_mlflow:
                fname = f"{plot_params_name[0]}_{plot_params_name[1]}_{title.replace(', ', '_').replace('=', '_')}.png"
                print(f"Logging figure to MLflow with name: {fname}")
            ax.plot()

def evaluate_classifier(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    predictions = model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Log Loss": log_loss(y_test, predictions_proba),
        "F1 Score": f1_score(y_test, predictions, average="weighted")
    }

    return metrics
