import numpy as np

__all__ = ['accuracy', 'precision', 'cross_entropy']

def accuracy(pred: np.ndarray, label: np.ndarray):
    return sum(pred==label) / len(pred)

def precision(pred: np.ndarray, label: np.ndarray):
    # TP_class_i / TP_class_i + FP_class_i
    ...

# technically not a metric
def cross_entropy(pred: np.ndarray, label: np.ndarray):
    ...

