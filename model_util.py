from joblib import dump, load
import pickle


class model_handler:
    def __init__(self):
        pass

    def save_model(self, model_obj, filename):
        _model = pickle.dumps(model_obj)
        _name = filename + '.joblib'
        dump(_model, _name)

    def load_model(self, filename):
        return load(filename)
