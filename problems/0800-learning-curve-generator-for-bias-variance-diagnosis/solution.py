import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    train_errors = []
    val_errors = []

    for n in train_sizes:
        X_train_poly = np.vander(X_train[:n].flatten(), N=degree + 1, increasing=True)
        X_val_poly = np.vander(X_val.flatten(), N=degree + 1, increasing=True)

        theta, _, _, _ = np.linalg.lstsq(X_train_poly, y_train[:n], rcond=None)

        train_err = np.mean((X_train_poly @ theta - y_train[:n]) ** 2)
        val_err = np.mean((X_val_poly @ theta - y_val) ** 2)
        train_errors.append(train_err)
        val_errors.append(val_err)

    final_train_error = train_errors[-1]
    final_val_error = val_errors[-1]
    diagnosis = ''
    if final_train_error > bias_threshold:
        diagnosis = 'high_bias'
    elif (final_val_error - final_train_error) > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'

    return {
        'train_errors' : train_errors,
        'val_errors' : val_errors,
        'diagnosis' : diagnosis
    }

