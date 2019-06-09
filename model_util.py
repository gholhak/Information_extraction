from joblib import dump, load
import pickle

"""
Save and load sklearn models to already defined relative path.

The class constructor defines a fixed path where the models are stored.

The positional arguments of the functions "save_model" and "load_model"
are merely the name of the classifier without any file extension. 
"""


class model_handler:
    def __init__(self):
        self.default_path = 'models\\'

    def save_model(self, model_obj, path):
        _model = pickle.dumps(model_obj)
        _name = self.default_path + path + '.joblib'
        dump(_model, _name)

    def load_model(self, filename):
        path_name = self.default_path + filename + '.joblib'
        _model = load(path_name)
        out_model = pickle.loads(_model)
        return out_model
