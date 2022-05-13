from types import AsyncGeneratorType
import numpy as np
try:
    import Image
except ImportError:
    from PIL import Image

class Compose:
    
    def __init__(self, transforms=[]):
        self.transforms = transforms
    
    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


# ================================================================================
# Transforms for NumPy ndarray
# ================================================================================
class Normalize:
    """Normalize a NumPy array with mean and standard deviation.

    Args:
        mean (float or sequence): mean for all values or sequence of means for each channel.
        std (float or sequence): 
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        return (array - mean) / std

class Flatten:
    ###Flatten a Numpy array.
    ###
    def __call__(self, array):
        return array.flatten()

class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
    
    def __call__(self, array):
        return array.astype(self.dtype)

ToFloat = AsType

