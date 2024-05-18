import tensorflow.keras.backend as K

def f1_score(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')  # Convert y_true to float32
    y_pred = K.cast(y_pred, 'float32') 
    """Compute the F1 score."""
    # Calculate true positives (tp), false positives (fp), and false negatives (fn)
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    # Calculate precision and recall
    precision = tp / (tp + fp + K.epsilon())  # Add epsilon to avoid division by zero
    recall = tp / (tp + fn + K.epsilon())

    print(precision, recall)

    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())

    return f1