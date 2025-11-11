import numpy as np

class StandardScaler1D:
    """Channel-wise standardization with invertibility.
    Expects data of shape (..., C) or (N, C) when fitting.
    """
    def __init__(self, eps: float = 1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, x: np.ndarray):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        return x * self.std_ + self.mean_
