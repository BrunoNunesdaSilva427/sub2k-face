
import numpy as np


class EigenfaceModel:
    def __init__(self, n_components=24):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  
        self.scale_ = None 

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
       
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[: self.n_components]

        
        projected = Xc @ self.components_.T
        self.scale_ = 127.0 / (np.abs(projected).max() + 1e-8)
        return self

    def project(self, X):
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def project_quantized(self, X):
        raw = self.project(X)
        q = np.round(raw * self.scale_)
        return np.clip(q, -127, 127).astype(np.int8)

    def save(self, path):
        np.savez(
            path,
            mean=self.mean_,
            components=self.components_,
            scale=self.scale_,
            n_components=self.n_components,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(n_components=int(data["n_components"]))
        model.mean_ = data["mean"]
        model.components_ = data["components"]
        model.scale_ = float(data["scale"])
        return model
