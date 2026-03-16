import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def train_random_forest(X_train, y_train):
    # Pipeline: fill any remaining NaN → scale → classify
    # This ensures imputation & scaling happen correctly inside every CV fold
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 4, 5, 6],
        "classifier__min_samples_split": [2, 3, 5],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )

    with joblib.parallel_config(prefer="threads"):
        grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_classifier(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)
