import abc

class Transformer():
    @abc.abstractmethod
    def __init__(self):
        super(Transformer, self).__init__()
        ...
    
    @abc.abstractmethod
    def transform():
        ...
    
    @abc.abstractmethod
    def inverse_transform():
        ...