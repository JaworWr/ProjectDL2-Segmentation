class BaseDataPreprocessing:
    def __init__(self, config):
        self.config = config

    def preprocess_train(self, datapoint):
        raise NotImplementedError()

    def preprocess_test(self, datapoint):
        return self.preprocess_train(datapoint)
