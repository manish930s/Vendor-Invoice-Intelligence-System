import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "predict_flag_invoice.pkl"

def load_model(model_path=MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict
  
    Returns
    -------
    """
    model = load_model()
    input_df = pd.DataFrame(input_data).astype(float)
    input_df["flag_invoice"] = model.predict(input_df)
    return input_df

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [15, 20],
        "invoice_dollars": [18500, 9000],
        "Freight": [320, 150],
        "total_item_quantity": [15, 20],
        "total_item_dollars": [18500, 9000]
    }
    predictions = predict_invoice_flag(sample_data)
    print(predictions)