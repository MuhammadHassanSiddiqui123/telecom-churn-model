# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to **churn** (cancel their service), so a retention team can contact at-risk customers before they leave.

The model is a scikit-learn pipeline (preprocessing + logistic regression) trained on the Telco customer churn dataset: 7,043 customers, 19 input features, about 26.5% of whom churned.

## Repository contents

| File | Description |
|---|---|
| `telco_churn.csv` | Raw dataset (7,043 rows, 21 columns) |
| `model.ipynb` | Notebook with data exploration, training and evaluation |
| `churn_model.pkl` | Trained scikit-learn `Pipeline`, ready for inference |
| `requirements.txt` | Python dependencies |

## Dataset

Each row is one customer. The target column is `Churn` (`Yes` / `No`).

| Group | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Identifier / target | `customerID` (not used for training), `Churn` |

**Data quality note:** `TotalCharges` is stored as text and has 11 blank values. All 11 belong to customers with `tenure = 0` (new customers who have not been billed yet). Convert the column to numeric and drop or fill those rows before modelling.

## Model

The saved model is a single `sklearn.pipeline.Pipeline`:

1. **Preprocessing (`ColumnTransformer`)**
   - Numeric (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`): `StandardScaler`
   - Categorical (the 15 remaining feature columns): `OneHotEncoder(drop="first", handle_unknown="ignore")`
2. **Classifier:** `LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000, random_state=42)`

`class_weight="balanced"` compensates for the imbalance (roughly 3 non-churners per churner), which favours catching churners over raw accuracy.

### Performance

Stratified 5-fold cross-validation of this exact configuration, on the 7,032 customers with a valid `TotalCharges`:

| Metric | Score |
|---|---|
| ROC-AUC | 0.845 |
| Accuracy | 0.746 |
| Recall (churners caught) | 0.798 |
| Precision | 0.515 |
| F1 | 0.626 |

Reading these numbers: the model finds about 80% of customers who will churn, but about half of the customers it flags would not have left. That trade-off suits low-cost retention offers. If outreach is expensive, raise the decision threshold above 0.5 to gain precision at the cost of recall.

## Quick start

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook model.ipynb
```

## Using the trained model

```python
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

# One row per customer, using the same 19 feature columns as the training data
# (everything except customerID and Churn)
df = pd.read_csv("telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

X = df.drop(columns=["customerID", "Churn"])
df["churn_probability"] = model.predict_proba(X)[:, 1]
df["churn_prediction"] = (df["churn_probability"] >= 0.5).astype(int)

print(df.sort_values("churn_probability", ascending=False)
        [["customerID", "churn_probability"]].head(10))
```

New data must contain all 19 feature columns with the same names and category labels. Unseen category values are ignored by the encoder instead of raising an error.

## Notes

- **Version compatibility:** the model was saved with scikit-learn 1.7.2. Use the pinned version in `requirements.txt` to avoid pickle compatibility warnings.
- **Security:** only load `.pkl` files from sources you trust. Unpickling can execute arbitrary code.
- **Scope:** the model uses a snapshot of customer attributes and has no time component, so it estimates risk but does not explain why a given customer might leave.

## License

Add a license of your choice (for example MIT) before sharing publicly, and check the terms of the original dataset source.
