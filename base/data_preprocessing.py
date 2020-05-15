class BaseDataPreprocessing:
    def __init__(self, config):
        self.config = config

    def preprocess_config(self):
        pass

    def preprocess(self, datapoint):
        raise NotImplementedError

    def preprocess_train(self, datapoint):
        return self.preprocess(datapoint)

    def preprocess_test(self, datapoint):
        return self.preprocess(datapoint)
