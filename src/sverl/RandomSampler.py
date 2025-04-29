import numpy as np 

class RandomSampler:
    """
    RandomSampler imputes missing features, given some observed features,
    by randomly selecting values from the same data sample.
    The class imputes **unconditioned** on the observed features.
    Args:
        data: A numpy array of shape (n, d), where n is the number of samples and d is the number of features.
        null_value: The value used to indicate missing features in the data.
    """
    def __init__(self, data, null_value=0):
        self.data = data
        self.size = data.shape[1] # Number of features d
        self.length = data.shape[0] # Number of samples n
        self.null_value = null_value


    def pred(self, observed_features, mask):
        """
        Predicts the missing features based on the mask.
        Args:
            mask: A binary mask indicating which features are missing (0) or present (1).
        Returns:
            A numpy array with the predicted values for the missing features.
        """

        pred = np.zeros(self.size)
        # Get missing features from the same data sample
        idx = np.random.randint(0, self.length)
        for i, b in enumerate(mask): 
            if b == self.null_value: 
                pred[i] = self.data[idx, i]
            else:
                pred[i] = observed_features[i]

        return pred

