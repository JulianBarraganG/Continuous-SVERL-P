import numpy as np 

class RandomSampler:
    def __init__(self, data):
        self.data = data
        self.size = data.shape[1]
        self.length = data.shape[0]


    def pred(self, observed_features, mask):
        """
        Predicts the missing features based on the mask.
        Args:
            mask: A binary mask indicating which features are missing (0) or present (1).
        Returns:
            A numpy array with the predicted values for the missing features.
        """

        pred = np.zeros(self.size)

        for i, b in enumerate(mask): 
            if b == 1: 
                #sample from the data
                idx = np.random.randint(0, self.length)
                pred[i] = self.data[idx, i]

        for i in range(self.size):
            if pred[i] == 0:
                pred[i] = observed_features[i]

        return pred
