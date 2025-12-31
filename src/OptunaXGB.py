# Optuna Optimization
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


RANDOM_STATE = 42

def objective(trial, X, y, preprocessor, scale_pos_weight):
    """
    Optuna objective function:
    - defines parameter search space
    - applies SMOTE inside each fold
    - performs stratified cross-validation
    - returns mean ROC AUC score
    """

    # Hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 1e-7, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 10.0, log=True),

        # Imbalance handling
        'scale_pos_weight': scale_pos_weight,

        # Fixed parameters
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
    }

    # Model
    model = XGBClassifier(**params)

    # Pipeline (same variable name)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_index, val_index in cv.split(X, y):

        # Extract folds
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Train pipeline
        pipeline.fit(X_train_fold, y_train_fold)

        # Predict probabilities
        y_pred_proba = pipeline.predict_proba(X_val_fold)[:, 1]

        # AUC score
        score = roc_auc_score(y_val_fold, y_pred_proba)
        scores.append(score)

    # Return mean CV score
    return np.mean(scores)
