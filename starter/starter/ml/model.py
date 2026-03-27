import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    rf_clss_model = RandomForestClassifier()
    rf_clss_model.fit(X_train, y_train)

    return rf_clss_model

def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def compute_slice_metrics(
    model : RandomForestClassifier,
    X,
    label,
    encoder,
    lb,
    categorical_features,
):
    """
    Compute model performance metrics on slices of the data for each categorical feature.

    Parameters
    ----------
    model : trained classifier
    X : pd.DataFrame
        Feature DataFrame BEFORE one-hot encoding
    label : str
        Name of label column
    encoder : OneHotEncoder
        The encoder used during training
    lb : LabelBinarizer
        Label binarizer used during training
    categorical_features : list
        List of categorical feature names

    Returns
    -------
    results : dict
        Mapping of: 
            feature -> category -> {precision, recall, fbeta}
    """

    slice_results = {}

    for feature in categorical_features:
        slice_results[feature] = {}
        categories = X[feature].unique()

        for cat in categories:
            # Filter rows where feature == category
            mask = X[feature] == cat
            X_slice = X[mask]

            # Skip tiny slices
            if len(X_slice) < 5:
                continue

            # Process the slice the same way as during training
            X_slice_processed, y_slice_processed, _, _ = process_data(
                X_slice,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = model.predict(X_slice_processed)
            prec, rec, fbeta = compute_model_metrics(y_slice_processed, preds)

            slice_results[feature][cat] = {
                "precision": float(prec),
                "recall": float(rec),
                "fbeta": float(fbeta),
                "n_samples": int(len(X_slice)),
            }

    return slice_results

def inference(model : RandomForestClassifier, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)
