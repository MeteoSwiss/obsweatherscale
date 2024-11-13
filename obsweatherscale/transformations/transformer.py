import abc


class Transformer():
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def transform(self, data):
        pass

    @abc.abstractmethod
    def inverse_transform(self, data):
        pass
