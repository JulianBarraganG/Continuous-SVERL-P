import numpy as np

class PCASampler:
    def __init__(self, data):
        self.mean = np.mean(data, axis=0)
        self.U, self.S, self.VT = np.linalg.svd(data - self.mean, full_matrices=False)

    def pred(self, observed_features, mask):
        # Note that "z" is a row vector, essentially z.T, so Vz = z.T @ VT
        # VT - the right singular vectors from the SVD
        # V - columns are the principal directions, i.e. this translates to the PCA space
        z = np.random.normal(size=self.VT.shape[0])
        x_sampled = self.mean + z @ self.VT
        return np.where(mask, observed_features, x_sampled)

