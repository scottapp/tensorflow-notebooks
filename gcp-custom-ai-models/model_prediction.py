import os
import pickle
import numpy as np


class CustomModelPrediction(object):
    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    def _postprocess(self, predictions):
        labels = ['github', 'nytimes', 'techcrunch']
        label_indexes = [np.argmax(prediction) for prediction in predictions]
        return [labels[label_index] for label_index in label_indexes]

    def predict(self, instances, **kwargs):
        preprocessed_data = self._processor.transform(instances)
        predictions =  self._model.predict(preprocessed_data)
        labels = self._postprocess(predictions)
        return labels

    @classmethod
    def from_path(cls, model_dir):
        import keras
        model = keras.models.load_model(os.path.join(model_dir, 'keras_saved_model.h5'))
        with open(os.path.join(model_dir, 'my_processor_state.pkl'), 'rb') as f:
            processor = pickle.load(f)
        return cls(model, processor)
