import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from modeling_evaluation import train_random_forest, evaluate_classifier
from data_preprocessing import load_invoice_data, apply_labels, split_data
from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

def main():
    # Load and label data
    df = load_invoice_data()
    df = apply_labels(df)

    # Split (NaN cleaning is handled inside split_data)
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # Train — Pipeline inside handles imputation + scaling + RF
    grid_search = train_random_forest(X_train, y_train)

    # Evaluate
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test,
        y_test,
        "Random Forest Classifier"
    )

    model_path = MODELS_DIR / 'predict_flag_invoice.pkl'
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\nModel saved at: {model_path}")

if __name__ == "__main__":
    main()
