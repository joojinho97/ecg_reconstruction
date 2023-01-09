from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return np.array(opt_threshold)


def class_rep(model, x, y):
    """Print classification report"""
    y_pred = model.predict(x)
    thresholds = get_optimal_precision_recall(y, y_pred)
    g = (y_pred > thresholds)
    print(classification_report(y, g, target_names=['LBBB', 'RBBB', 'AF'], zero_division=0))


if __name__ == "__main__":
    """Predict and performance check"""
    model = load_model('./model.hdf5', compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())

    # x = "Load x data"
    # y = "Load y data"

    class_rep(model, x, y)
