from abc import ABCMeta, abstractmethod

class Util_Interface():
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self, img_file: str):
        '''
        img_file: str, file path of input image.
        returns: prediction, 2-dementional numpy.ndarray, type of np.uint8.
        '''
        pass