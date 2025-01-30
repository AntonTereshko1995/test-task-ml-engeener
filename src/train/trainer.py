from catboost import CatBoostClassifier
import numpy as np
from types import SimpleNamespace
from typing import Dict, Any, List
from sklearn.model_selection import GridSearchCV
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
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
    model = CatBoostClassifier(**vars(params.model.baseline))

    print(f"Looking for best combination of parameters with objective {params.model.tune.metric}. Parameters are {params.model.tune.search.to_dict().keys()}")
    search = fit_and_optimize(X_train, 
                              y_train, 
                              base_model=model,
                              param_grid=params.model.tune.search.to_dict(),
                              cv=params.model.tune.cv,
                              scoring_fit=params.model.tune.metric)

    print(f"Evaluating the performance of the model")
    evaluator.evaluate_cat_boost_search(search, plot_params_name=['learning_rate', 'depth'], to_mlflow = True)

    model = search.best_estimator_
    metrics = evaluator.evaluate_cat_boost_classifier(model, X_test, y_test)
    print(f"Logging the pipeline to MLflow")
    print(search.best_params_)
    print(metrics)

    return model

def train_bert_classifier(tokenizer, train_dataset, valid_dataset, test_dataset, num_labels, device, params: SimpleNamespace) -> Any:
    model = BertForSequenceClassification.from_pretrained(
        params.model.baseline.model_name, 
        num_labels=num_labels, 
        problem_type=params.model.baseline.problem_type)
    
    model.to(device)
    model.config.hidden_dropout_prob = 0.3
    model.config.attention_probs_dropout_prob = 0.3

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        save_strategy=params.model.baseline.save_strategy,
        per_device_train_batch_size=int(params.model.baseline.batch_size),  # Ensure batch size is int
        per_device_eval_batch_size=int(params.model.baseline.batch_size),
        learning_rate=float(params.model.baseline.learning_rate),
        num_train_epochs=int(params.model.baseline.num_train_epochs),
        weight_decay=float(params.model.baseline.weight_decay),
        logging_steps=int(params.model.baseline.logging_steps),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset 
    )

    trainer.train()

    evaluator.evaluate_bert(trainer, test_dataset)
    return (model, trainer)