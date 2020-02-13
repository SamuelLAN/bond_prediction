import numpy as np


class MultiLabelClassification:
    @staticmethod
    def accuracy(y_true, y_pred, mask=None):
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        if isinstance(mask, type(None)):
            return np.mean(y_true == y_pred)

        return np.sum(np.cast[np.float32](y_true == y_pred) * mask) / (np.sum(mask) * y_true.shape[0])

    @staticmethod
    def precision(y_true, y_pred, mask=None, threshold=0.5):
        if not isinstance(mask, type(None)):
            y_pred = y_pred * mask
            y_true = y_true * mask

        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        true_positive = np.sum(y_true * y_pred)
        pred_true_num = np.sum(y_pred)
        return true_positive / (pred_true_num + 0.0001)

    @staticmethod
    def recall(y_true, y_pred, mask=None, threshold=0.5):
        if not isinstance(mask, type(None)):
            y_pred = y_pred * mask
            y_true = y_true * mask

        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        true_positive = np.sum(y_true * y_pred)
        y_true_num = np.sum(y_true)
        return true_positive / (y_true_num + 0.0001)

    @staticmethod
    def f1(y_true, y_pred, mask=None, threshold=0.5):
        precision = MultiLabelClassification.precision(y_true, y_pred, mask, threshold)
        recall = MultiLabelClassification.recall(y_true, y_pred, mask, threshold)
        return 2 * (precision * recall) / (precision + recall + 0.0001)

    @staticmethod
    def hamming_loss(y_true, y_pred, mask=None, threshold=0.5):
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        y_not_pred = np.cast[np.int32](y_pred == 0)
        y_not_true = np.cast[np.int32](y_true == 0)

        if isinstance(mask, type(None)):
            return np.mean(np.cast[np.float32](y_not_true * y_pred + y_true * y_not_pred))
        return np.sum(np.cast[np.float32](y_not_true * y_pred + y_true * y_not_pred) * mask) / (
                    np.sum(mask) * y_true.shape[0])
