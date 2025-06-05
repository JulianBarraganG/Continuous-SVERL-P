import numpy as np

class UnifSampler:
    """
    Samples from the state space uniformly at random,
    except it is constrained by the ranges provided.
    These ranges are taken from 
    """
    def __init__(self, ranges: list[tuple[float, float]], null_value=0):
        self.ranges = ranges
        self.size = len(ranges) # Number of features d
        self.null_value = null_value
    
    def sampler(self, min_val, max_val):
        type_checker = min_val
        if isinstance(type_checker, float):
            return np.random.uniform(min_val, max_val)
        if isinstance(type_checker, bool) or isinstance(type_checker, int):
            return np.random.random_integers(min_val, max_val)
        if isinstance(type_checker, str):
            if type_checker == "ninf":
                return np.random.normal()
        # Missing (ninf, b] and [a, inf) sampling, which other envs cover.
        # See: https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box.sample

    def pred(self, observed_features, mask):
        """
        Pred fnc
        """

        #TODO: Speed up this function, by chaching the sampling methods.
        pred = np.zeros(self.size, dtype=np.float32)
        for i, b in enumerate(mask):
            if b == self.null_value:
                min_val, max_val = self.ranges[i]
                imputed_feature_i = self.sampler(min_val, max_val)
                pred[i] = imputed_feature_i
            else:
                pred[i] = observed_features[i]

        return pred

