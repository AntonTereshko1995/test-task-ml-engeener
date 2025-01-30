
import sklearn_evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, log_loss, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import torch

def evaluate_cat_boost_search(search: GridSearchCV, plot_params_name: List[str], subset_params_name: List[str] = None, to_mlflow: bool = False):
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

def evaluate_cat_boost_classifier(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    predictions = model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Log Loss": log_loss(y_test, predictions_proba),
        "F1 Score": f1_score(y_test, predictions, average="weighted")
    }
    return metrics

def evaluate_bert(trainer, test_dataset):
    results = trainer.evaluate(test_dataset)
    predictions = trainer.predict(test_dataset)
    probs = torch.sigmoid(torch.tensor(predictions.predictions))  # Convert logits to probabilities
    preds = (probs > 0.5).int().numpy()  # Apply threshold of 0.5
    labels = np.array(predictions.label_ids)  # Convert labels to numpy array

    metrics = {
        "Accuracy (Subset)": accuracy_score(labels, preds),  # Exact match accuracy
        "Log Loss": log_loss(labels, probs.numpy()),  # Lower is better
        "F1 Score (Macro)": f1_score(labels, preds, average="macro"),  # F1 across all labels
        "F1 Score (Micro)": f1_score(labels, preds, average="micro"),
        "Precision (Macro)": precision_score(labels, preds, average="macro"),
        "Recall (Macro)": recall_score(labels, preds, average="macro"),
        "Hamming Loss": hamming_loss(labels, preds),  # Penalizes incorrect labels
        "Jaccard Score (Macro)": jaccard_score(labels, preds, average="macro"),
    }
    
    df_results = pd.DataFrame([metrics])
    print(df_results.to_string(index=False))

def debug_predict_and_evaluate(model, device, tokenizer, data_frame, texts):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.to("cpu").numpy()
    probs = torch.sigmoid(torch.tensor(logits))  # Convert logits to probabilities
    preds = (probs > 0.5).int().numpy()  # Threshold at 0.5
    
    real_labels = data_frame[data_frame["Title"].isin(texts)][["Title", "Features"]]
    # Print real labels
    print(real_labels)

    print("Raw Logits:", logits)
    print("Probabilities:", probs.numpy())
    print("Binary Predictions:", preds)

    metrics = bert_predict_and_evaluate(model, tokenizer, texts, real_labels)
    print(metrics)

    return preds

def bert_predict_and_evaluate(model, tokenizer, texts, true_labels):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.to("cpu").numpy()  # Move logits to CPU for processing
    metrics = bert_compute_metrics(logits, true_labels)  # Compute evaluation metrics
    print(metrics)
    return metrics

def bert_compute_metrics(predictions, true_labels):
    probs = torch.sigmoid(torch.tensor(predictions))
    preds = (probs > 0.5).int().numpy()
    true_labels = np.array(true_labels)  # Ensure true labels are numpy array
    metrics = {
        "Accuracy (Subset)": accuracy_score(true_labels, preds),  # Subset accuracy (exact match)
        "Log Loss": log_loss(true_labels, probs.numpy()),  # Log loss (lower is better)
        "F1 Score (Macro)": f1_score(true_labels, preds, average="macro"),  # F1-score across all labels
        "F1 Score (Micro)": f1_score(true_labels, preds, average="micro"),
        "Precision (Macro)": precision_score(true_labels, preds, average="macro"),
        "Recall (Macro)": recall_score(true_labels, preds, average="macro"),
        "Hamming Loss": hamming_loss(true_labels, preds),  # Penalizes incorrect labels
        "Jaccard Score (Macro)": jaccard_score(true_labels, preds, average="macro")
    }
    return metrics