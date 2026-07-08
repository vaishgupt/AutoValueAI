import os
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Save any Python object (model, preprocessor, etc.) to disk.
    """

    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)


def load_object(file_path):
    """
    Load saved object.
    """

    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params
):
    """
    Train multiple models using GridSearchCV
    and return model scores along with the best model.
    """

    report = {}

    for model_name in models:

        model = models[model_name]
        param = params[model_name]

        gs = GridSearchCV(
            estimator=model,
            param_grid=param,
            cv=3,
            scoring="r2",
            n_jobs=-1,
            verbose=1
        )

        gs.fit(X_train, y_train)

        # Train model with best parameters
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        score = r2_score(y_test, predictions)

        report[model_name] = score

        print(f"{model_name} Best Parameters: {gs.best_params_}")

    # Find the best model AFTER evaluating all models
    best_model_name = max(report, key=report.get)
    best_model = models[best_model_name]

    print("\nBest Model:", best_model_name)
    print(f"Best R² Score: {report[best_model_name]:.4f}")

    return report, best_model