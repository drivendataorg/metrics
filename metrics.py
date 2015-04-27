import numpy as np

# Defined for your convenience; these are the
# class_column_indices for the Box-Plots for Education competition
# www.drivendata.org/competitions/4/
BOX_PLOTS_COLUMN_INDICES = [range(37),
                            range(37,48),
                            range(48,51),
                            range(51,76),
                            range(76,79),
                            range(79,82),
                            range(82,87),
                            range(87,96),
                            range(96,104)]


# defined for your convenience; these are the penalties
# for the Keeping it Clean competition
KEEPING_IT_CLEAN_WEIGHTS = np.array([1., 2., 5.], dtype=np.float64)

def multi_multi_log_loss(predicted, actual, class_column_indices, eps=1e-15):
    """Multi class, multi-label version of Logarithmic Loss metric.

    :param predicted: a 2d numpy array of the predictions that are probabilities [0, 1]
    :param actual: a 2d numpy array of the same shape as your predictions. 1 for the actual labels, 0 elsewhere 
    :return: The multi-multi log loss score for this set of predictions
    """
    class_scores = np.ones(len(class_column_indices), dtype=np.float64)

    # calculate log loss for each set of columns that belong to a class:
    for k, this_class_indices in enumerate(class_column_indices):
        # get just the columns for this class
        preds_k = predicted[:, this_class_indices]

        # normalize so probabilities sum to one (unless sum is zero, then we clip)
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)

        actual_k = actual[:, this_class_indices]

        # shrink predictions
        y_hats = np.clip(preds_k, eps, 1 - eps)
        sum_logs = np.sum(actual_k * np.log(y_hats))
        class_scores[k] = (-1.0 / actual.shape[0]) * sum_logs

    return np.average(class_scores)


def weighted_rmsle(predicted, actual, weights=None):
        """ Calculates RMSLE weighted by a vector of weights.

        :param predicted: the predictions
        :param actual: the actual true data
        :param weights: how "important" each column is (if None, assume equal)
        :return: WRMSLE
        """
        # force floats
        predicted = predicted.astype(np.float64)
        actual = actual.astype(np.float64)

        # if no weights, assume equal weighting
        if weights is None:
            weights = np.ones(predicted.shape[1], dtype=np.float64)

        # reshape as a column matrix
        weights = weights.reshape(-1, 1).astype(np.float64)

        # make sure that there are the right number of weights
        if weights.shape[0] != predicted.shape[1]:
            error_msg = "Weight matrix {} must have same number of entries as columns in predicted ({})."
            raise Exception(error_msg.format(weights.shape, predicted.shape[1]))

        # calculate weighted scores
        predicted_score = predicted.dot(weights)
        actual_score = actual.dot(weights)

        # calculate log error
        log_errors = np.log(predicted_score + 1) - np.log(actual_score + 1)

        # return RMSLE
        return np.sqrt((log_errors ** 2).mean())

