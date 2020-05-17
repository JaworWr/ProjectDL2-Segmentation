class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self):
        if self.model is None:
            raise RuntimeError("You have to build the model before saving it.")

        if self.config.save_checkpoint:
            print("Saving model...")
            self.model.save_weights(self.config.save_checkpoint)
            print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self):
        if self.model is None:
            raise RuntimeError("You have to build the model before loading it.")

        if self.config.load_checkpoint:
            print(f"Loading model checkpoint {self.config.load_checkpoint}...")
            self.model.load_weights(self.config.load_checkpoint)
            print("Model loaded")

    def build(self):
        raise NotImplementedError

    def compile(self):
        raise NotImplementedError

    def __getattr__(self, item):
        if self.model is None:
            raise RuntimeError("You have to build the model first.")
        return getattr(self.model, item)
