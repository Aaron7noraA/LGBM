import lightgbm as lgb
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load example dataset
X, y = load_iris(return_X_y=True)
num_classes = len(set(y))
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define objective function
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    model = lgb.train(params, dtrain, valid_sets=[dvalid], 
                      num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)

    y_pred = model.predict(X_valid).argmax(axis=1)
    return accuracy_score(y_valid, y_pred)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best result
print("Best Accuracy:", study.best_value)
print("Best Parameters:", study.best_params)





import optuna.visualization as vis
from optuna.visualization._plotly_imports import go

# Optimization history
fig1 = vis.plot_optimization_history(study)
fig1.write_image("optuna_optimization_history.png")

# Parameter importance
fig2 = vis.plot_param_importances(study)
fig2.write_image("optuna_param_importance.png")



