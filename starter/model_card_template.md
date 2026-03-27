# Model Card: RandomForest Classifier for Census Income Prediction

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## 1. Model Overview
This model is part of the **Scalable ML Pipeline** project and is built using a **RandomForestClassifier**. The goal of the model is to predict whether an individual's annual income exceeds $50K based on demographic and employment attributes.

The model is fully integrated into an MLOps pipeline, including:
- Data ingestion and cleaning
- Feature engineering and encoding
- Model training
- Evaluation and slice evaluation
- Deployment via FastAPI

## 2. Intended Use
The model is intended to **predict income category (`<=50K` or `>50K`)** for adult individuals using U.S. Census data. It supports decision-making in:
- Education analysis
- Resource allocation research
- Socioeconomic studies

## 3. Training Data
The model was trained on the **UCI Adult Census dataset**. The dataset includes the following columns:
- Numerical: `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- Target: `salary`

Data preprocessing steps included:
- Stripping whitespace
- Handling missing values
- One-hot encoding for categorical fields
- Train-test split (80/20)

## 4. Model Details
### Algorithm
- **RandomForestClassifier** from scikit‑learn
- Hyperparameters:
  - n_estimators = 100 (default)
  - max_depth = None
  - random_state = 42

### Why RandomForest?
- Handles non-linear relationships
- Robust to outliers
- Works well with mixed categorical + numerical data
- Low risk of overfitting compared to single decision trees

## 5. Evaluation Metrics
The model was evaluated on the test set using:
- **Precision**
- **Recall**
- **FBeta-score**

Slice evaluation was also performed on categorical slices such as:
- `sex`
- `race`
- `education`
- `relationship`

These help identify **fairness issues** across demographic segments.

## 6. Ethical Considerations
The model is trained on census data that may reflect societal biases. Potential risks include:
- Underrepresentation of minority groups
- Embedded socioeconomic inequalities
- Biased predictions for certain demographic slices

Mitigations include:
- Slice-based performance reporting
- Transparent model card documentation
- Advising users not to deploy in sensitive scenarios

## 7. Limitations
- Predictions are only as good as the training data
- Dataset is from 1994 and may not reflect current demographics
- Income is influenced by many unobserved factors (skills, experience, location)
- Cannot generalize outside U.S. population

## 8. Versioning
- Model Version: 1.0.0
- Pipeline Version: 1.0.0
- Data Version: census_cleaned_v1

## 9. How to Use
### Loading the model
The model and preprocessors (encoder, label binarizer) are stored in the `model/` directory.

Example:
```python
import pickle
model = pickle.load(open('model/rf_clss_model.pkl','rb'))
encoder = pickle.load(open('model/encoder.pkl','rb'))
lb = pickle.load(open('model/lbinarizer.pkl','rb'))
```

### Sending inference requests (FastAPI)
```bash
curl -X POST "http://localhost:8000/infer"     -H "Content-Type: application/json"     -d '{
        "age": 39,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }'
```

## 10. Future Improvements
- Hyperparameter tuning (GridSearch / Optuna)
- Better fairness evaluation with Aequitas
- Updating dataset to modern census records
- Model compression for faster inference

---
This model card documents the behavior and context of the RandomForest model as part of a scalable ML pipeline.
